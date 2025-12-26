"""用于比较 Fast-dLLM 与原生 Qwen2.5 的注意力表现。

该脚本会加载两个模型，在同一个提示上运行短解码，收集注意力权重，
并将可比的窗口保存为 PNG/GIF。它复用已补丁的 ``modeling.py`` 中的
Fast-dLLM 跟踪钩子，同时利用原生 Qwen2.5 模型的 ``output_attentions``。
"""

import os
import time
import types
from typing import Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import generation_functions

matplotlib.rcParams["font.family"] = "DejaVu Sans Mono"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans Mono", "DejaVu Sans"]

# 关闭 flash / sdpa，以确保使用补丁后的 Fast-dLLM 注意力路径。
os.environ.setdefault("PYTORCH_USE_FLASH_ATTENTION", "0")
os.environ.setdefault("PYTORCH_ENABLE_SDPA", "0")
os.environ.setdefault("FLEX_ATTENTION_DISABLE", "0")

# 强制原生 Qwen 使用 eager attention，以便 ``output_attentions`` 生效。
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


DEFAULT_LAYERS = [0]
# DEFAULT_LAYERS = [4,5]


def _format_with_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    """若分词器提供聊天模板，则按聊天格式包装提示，否则回退原文。"""

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:  # pragma: no cover - 仅防御性日志
            print(f"[ChatTemplate] fallback to raw prompt: {exc}")

    return prompt


def _default_layers(num_layers: int) -> List[int]:
    """返回用于跟踪的默认层索引（裁剪到模型深度）。"""
    return sorted({idx for idx in DEFAULT_LAYERS if idx < num_layers})


def _clear_fastdllm_traces(model: torch.nn.Module) -> None:
    """就地重置 Fast-dLLM 的注意力/词元跟踪。"""
    for layer in model.model.layers:
        sa = layer.self_attn
        if hasattr(sa, "_attn_trace"):
            sa._attn_trace.clear()
        if hasattr(sa, "_last_attn"):
            sa._last_attn = None
        if hasattr(sa, "_attn_tokens"):
            sa._attn_tokens.clear()


def _save_attention_blocks(
    traces: Sequence[Sequence[torch.Tensor]],
    token_traces: Sequence[Sequence[torch.Tensor]],
    tokenizer: AutoTokenizer,
    save_dir: str,
    save_layers: Iterable[int],
    window: int = 32,
    make_gif: bool = True,
    keep_stride: Optional[int] = None,
) -> None:
    """为指定的层保存逐步注意力热力图并拼接 GIF。

    参数：
        traces：按层组织的注意力张量列表。
        token_traces：与 ``traces`` 对齐的按层词元 ID 张量列表。
        tokenizer：用于解码查询词元、生成纵轴标签的分词器。
        save_dir：输出制品的根目录。
        save_layers：需要持久化的层。
        window：查询/键的窗口大小。
        make_gif：是否将 PNG 拼成 GIF（仅 Fast-dLLM 使用）。
        keep_stride：若设置，仅保留满足 ``step_id % keep_stride == keep_stride - 1`` 的 PNG，
            以便对长自回归序列下采样（用于 Qwen2.5）。
    """
    os.makedirs(save_dir, exist_ok=True)
    save_layers = set(save_layers)

    for layer_id, (trace, tok_trace) in enumerate(zip(traces, token_traces)):
        if layer_id not in save_layers:
            continue

        if not trace:
            print(f"Layer {layer_id}: empty trace, skip saving")
            continue

        layer_dir = os.path.join(save_dir, f"layer_{layer_id:02d}")
        # 移除上一次运行遗留的 PNG/GIF，保持步索引对齐。
        if os.path.isdir(layer_dir):
            for fname in os.listdir(layer_dir):
                if fname.startswith("step_") and fname.endswith(".png"):
                    os.remove(os.path.join(layer_dir, fname))
                elif fname.startswith("block_") and fname.endswith(".gif"):
                    os.remove(os.path.join(layer_dir, fname))
        os.makedirs(layer_dir, exist_ok=True)

        last_step = len(trace) - 1

        for step_id, attn in enumerate(trace):
            if (
                keep_stride is not None
                and step_id % keep_stride != keep_stride - 1
                and step_id != last_step
            ):
                continue

            attn_head0 = attn[0, 0]  # 形状：[L_q, L_k]
            Lq, Lk = attn_head0.shape

            if step_id >= len(tok_trace):
                print(f"  step {step_id}: missing token trace, skip")
                continue

            step_token_ids = tok_trace[step_id][0]  # 形状：[L_q]
            if step_token_ids.shape[0] != Lq:
                print(
                    f"  step {step_id}: token length {step_token_ids.shape[0]} != L_q {Lq}, skip",
                )
                continue

            q_start = max(Lq - window, 0)
            q_end = Lq - 1
            k_start = max(Lk - window, 0)
            k_end = Lk - 1

            attn_block = attn_head0[q_start : q_end + 1, k_start : k_end + 1]
            # Matplotlib 无法渲染 bfloat16 张量；绘图前转换为 float32。
            attn_block = attn_block.to(dtype=torch.float32)

            q_ids = step_token_ids[q_start : q_end + 1].tolist()
            q_tokens = [tokenizer.decode([tid]) for tid in q_ids]

            x_pos = list(range(attn_block.shape[1]))
            y_pos = list(range(attn_block.shape[0]))
            y_labels = [f"{q_start + i}:{q_tokens[i]}" for i in range(len(q_tokens))]

            plt.figure(figsize=(12, 8))
            plt.imshow(attn_block, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(
                f"Layer {layer_id} - Step {step_id}\n"
                f"Q[{q_start}~{q_end}] vs K[{k_start}~{k_end}]",
            )
            plt.yticks(y_pos, y_labels, fontsize=6)
            plt.ylabel("Query tokens (windowed)")
            plt.xlabel("Key indices (windowed)")
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, f"step_{step_id:03d}.png"), dpi=140)
            plt.close()

        if not make_gif:
            continue

        # 按连续的 L_k 段拼接 GIF（便于观察块切换）。
        png_files = sorted(
            f
            for f in os.listdir(layer_dir)
            if f.startswith("step_") and f.endswith(".png")
        )
        if not png_files:
            print(f"Layer {layer_id}: no PNGs, skip GIF")
            continue

        prev_Lk: Optional[int] = None
        block_id = 0
        frames: List[torch.Tensor] = []

        for png in png_files:
            step_id = int(png.split("_")[1].split(".")[0])
            if step_id >= len(trace):
                print(f"  step {step_id}: missing trace entry, skip GIF frame")
                continue

            attn = trace[step_id][0, 0]
            Lk = attn.shape[1]

            if prev_Lk is None:
                prev_Lk = Lk
            elif Lk != prev_Lk:
                gif_path = os.path.join(layer_dir, f"block_{block_id:02d}.gif")
                imageio.mimsave(gif_path, frames, duration=0.2)
                print(f"Layer {layer_id}: saved {gif_path} ({len(frames)} frames)")
                block_id += 1
                frames = []
                prev_Lk = Lk

            frames.append(imageio.imread(os.path.join(layer_dir, png)))

        if frames:
            gif_path = os.path.join(layer_dir, f"block_{block_id:02d}.gif")
            imageio.mimsave(gif_path, frames, duration=0.2)
            print(f"Layer {layer_id}: saved {gif_path} ({len(frames)} frames)")


def run_fastdllm_trace(
    prompt: str,
    save_dir: str,
    save_layers: Optional[Iterable[int]] = None,
    block_size: int = 32,
    small_block_size: int = 8,
    max_new_tokens: int = 1024,
    threshold: float = 0.9,
) -> None:
    """运行 Fast-dLLM 解码并持久化捕获的注意力图。"""
    model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Fast-dLLM] Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
        model,
    )

    _clear_fastdllm_traces(model)

    formatted_prompt = _format_with_chat_template(tokenizer, prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = torch.tensor([inputs.shape[1]], device=device)

    print("[Fast-dLLM] Running diffusion decoding…")
    t0 = time.time()
    generated = model.mdm_sample(
        inputs,
        tokenizer=tokenizer,
        block_size=block_size,
        max_new_tokens=max_new_tokens,
        small_block_size=small_block_size,
        min_len=min(inputs.shape[1], block_size - 1),
        seq_len=seq_len,
        mask_id=151665,
        stop_token=151645,
        threshold=threshold,
    )
    t1 = time.time()

    generated_ids = generated[0]
    print(
        f"[Fast-dLLM] Finished in {t1 - t0:.2f}s; new tokens: {generated_ids.shape[0] - inputs.shape[1]}"
    )

    layer_traces = []
    layer_tokens = []
    for layer in model.model.layers:
        sa = layer.self_attn
        layer_traces.append(getattr(sa, "_attn_trace", []))
        layer_tokens.append(getattr(sa, "_attn_tokens", []))

    layers_to_save = (
        set(save_layers)
        if save_layers is not None
        else set(_default_layers(model.config.num_hidden_layers))
    )

    _save_attention_blocks(
        layer_traces,
        layer_tokens,
        tokenizer,
        os.path.join(save_dir, "fastdllm"),
        save_layers=layers_to_save,
    )

    print(
        "[Fast-dLLM] Saved attention PNG/GIFs under",
        os.path.abspath(os.path.join(save_dir, "fastdllm")),
    )
    print(
        "[Fast-dLLM] Sampled text:\n",
        tokenizer.decode(generated_ids[inputs.shape[1] :], skip_special_tokens=True),
    )


def run_qwen2p5_trace(
    prompt: str,
    save_dir: str,
    save_layers: Optional[Iterable[int]] = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens: int = 1024,
    window: int = 32,
) -> None:
    """Run vanilla Qwen2.5 and save attention windows for comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Qwen2.5] Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    # 确保生成阶段同样使用 eager attention，以便 ``output_attentions`` 生效。
    if hasattr(model, "generation_config"):
        model.generation_config.attn_implementation = "eager"

    formatted_prompt = _format_with_chat_template(tokenizer, prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    print("[Qwen2.5] Running generate with attention capture…")
    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_attentions=True,
        do_sample=False,
    )
    t1 = time.time()

    sequences = outputs.sequences  # 形状：[1, prompt + generated]
    prompt_len = inputs["input_ids"].shape[1]
    print(f"[Qwen2.5] Finished in {t1 - t0:.2f}s; total length: {sequences.shape[1]}")

    # 生成阶段的注意力只包含最新 token（L_q = 1），热力图会呈现细条纹。
    # 为了得到可与 Fast-dLLM 对比的网格效果，重新对完整序列做一次前向，
    # 以获取全序列的注意力。
    with torch.no_grad():
        full_outputs = model(
            sequences,
            attention_mask=torch.ones_like(sequences),
            output_attentions=True,
            use_cache=False,
        )

    layer_traces: List[List[torch.Tensor]] = []
    token_traces: List[List[torch.Tensor]] = []

    # 将全序列注意力按步切片成窗口，使保存的热力图呈现多帧效果，
    # 而不是单一静态网格，从而与 Fast-dLLM 可视化一致。
    for attn in full_outputs.attentions:  # 形状：[1, H, L, L]
        per_layer: List[torch.Tensor] = []
        per_layer_tokens: List[torch.Tensor] = []

        for end in range(1, sequences.shape[1] + 1):
            per_layer.append(attn[:, :, :end, :end].detach().cpu())
            per_layer_tokens.append(sequences[:, :end].detach().cpu())

        layer_traces.append(per_layer)
        token_traces.append(per_layer_tokens)

    layers_to_save = (
        set(save_layers)
        if save_layers is not None
        else set(_default_layers(model.config.num_hidden_layers))
    )

    _save_attention_blocks(
        layer_traces,
        token_traces,
        tokenizer,
        os.path.join(save_dir, "qwen2_5"),
        save_layers=layers_to_save,
        window=window,
        make_gif=False,
        keep_stride=32,
    )

    print(
        "[Qwen2.5] Saved attention PNG/GIFs under",
        os.path.abspath(os.path.join(save_dir, "qwen2_5")),
    )

    generated_text = tokenizer.decode(
        sequences[0][prompt_len:], skip_special_tokens=True
    )
    print("[Qwen2.5] Sampled text:\n", generated_text)


if __name__ == "__main__":
    SAMPLE_PROMPT = (
        "Josh decides to flip a house. He buys a house for $80,000 and then puts in $50,000 in repairs. "
        "This increased the value of the house by 150%. How much profit did he make?",
    )

    # 默认需要跟踪的层；如果想使用不同索引，可在此修改列表。
    SAVE_LAYERS = DEFAULT_LAYERS
    SAVE_DIR = "attn_compare"

    run_fastdllm_trace(SAMPLE_PROMPT, SAVE_DIR, save_layers=SAVE_LAYERS)
    run_qwen2p5_trace(SAMPLE_PROMPT, SAVE_DIR, save_layers=SAVE_LAYERS)
