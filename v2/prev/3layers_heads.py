import os
import time
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import types
import generation_functions

matplotlib.rcParams['font.family'] = 'DejaVu Sans Mono'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans Mono', 'DejaVu Sans']

# 禁用 flash / sdpa 以便使用我们 patch 的实现
os.environ["PYTORCH_USE_FLASH_ATTENTION"] = "0"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"
os.environ.setdefault("FLEX_ATTENTION_DISABLE", "0")

# ===========================================================
# Load model
# ===========================================================

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device)
model.eval()

print("Model Loaded. Device:", next(model.parameters()).device)

# 绑定 diffusion 采样函数
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
    model
)

# ===========================================================
# Clear old attention traces
# ===========================================================

def clear_all_attn_traces(model):
    for layer in model.model.layers:
        sa = layer.self_attn
        if hasattr(sa, "_attn_trace"):
            sa._attn_trace.clear()
        if hasattr(sa, "_last_attn"):
            sa._last_attn = None
        if hasattr(sa, "_attn_tokens"):
            sa._attn_tokens.clear()

clear_all_attn_traces(model)
print("Cleared previous attention traces.")

# ===========================================================
# Run diffusion decoding
# ===========================================================

prompt = ("Nick is choosing between two jobs. Job A pays $15 an hour for 2000 hours a year, and is in a state with a 20% total tax rate. Job B pays $42,000 a year and is in a state that charges $6,000 in property tax and a 10% tax rate on net income after property tax. How much more money will Nick make at the job with a higher net pay rate, compared to the other job?")

inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
print("\nRunning diffusion decoding...")
print("Input token length:", inputs.shape[1], "\n")

t0 = time.time()

generated = model.mdm_sample(
    inputs,
    tokenizer=tokenizer,
    block_size=32,
    max_new_tokens=2048,
    small_block_size=8,
    min_len=min(inputs.shape[1], 31),
    seq_len=torch.tensor([inputs.shape[1]], device=device),
    mask_id=151665,
    stop_token=151645,
    threshold=0.9,
)

t1 = time.time()

print(f"Diffusion finished in {t1 - t0:.3f} seconds")
generated_ids = generated[0]
print("Generated tokens:", generated_ids.shape[0] - inputs.shape[1])

# ===========================================================
# Collect attention traces & token traces from all layers
# ===========================================================

print("\nCollecting attention traces...")

layer_traces = []
layer_tokens = []

for i, layer in enumerate(model.model.layers):
    sa = layer.self_attn
    trace = getattr(sa, "_attn_trace", [])
    tokens = getattr(sa, "_attn_tokens", [])

    # 保证长度一致（防御用，正常应当相同）
    if len(trace) != len(tokens):
        print(f"WARNING: layer {i} trace len {len(trace)} != token len {len(tokens)}")

    layer_traces.append(trace)
    layer_tokens.append(tokens)

    print(f"Layer {i}: {len(trace)} attention steps")

max_steps = max(len(t) for t in layer_traces)
print("\nMax attention steps captured:", max_steps)

if max_steps == 0:
    print("ERROR: No attention captured. Check modeling.py patch.")
    raise SystemExit

# ===========================================================
# Save attention maps：Q = 当前 forward 的 noisy block xt
# ===========================================================

save_layers = {0}
save_dir = "diffusion_attn"
os.makedirs(save_dir, exist_ok=True)

WINDOW = 32  # 我们希望看的 Query 窗口大小 = block_size

print("\nSaving attention maps (Q = current noisy block xt)...")

for layer_id, (trace, tok_trace) in enumerate(zip(layer_traces, layer_tokens)):

    if layer_id not in save_layers:
        continue

    print(f"\nLayer {layer_id} -------------------------------")

    layer_dir = os.path.join(save_dir, f"layer_{layer_id:02d}")
    os.makedirs(layer_dir, exist_ok=True)

    for step_id, attn in enumerate(trace):
        attn_head0 = attn[0, 0]  # [L_q, L_k]
        Lq, Lk = attn_head0.shape

        if step_id >= len(tok_trace):
            print(f"  step {step_id}: no token trace, skip.")
            continue

        # 当前 step 的输入 token 序列（通常就是这一 step 的 xt block）
        step_token_ids = tok_trace[step_id]  # [1, L_q]
        step_token_ids = step_token_ids[0]   # [L_q]

        if step_token_ids.shape[0] != Lq:
            print(f"  step {step_id}: token_len {step_token_ids.shape[0]} != L_q {Lq}, skip.")
            continue

        # -------- Q：最后 WINDOW 个 query = 当前 noisy block 尾部 ----------
        if Lq <= WINDOW:
            q_start = 0
            q_end = Lq - 1
        else:
            q_end = Lq - 1
            q_start = Lq - WINDOW
        wq = q_end - q_start + 1

        # -------- K：最后 WINDOW 个 key，用索引即可 ----------
        if Lk <= WINDOW:
            k_start = 0
            k_end = Lk - 1
        else:
            k_end = Lk - 1
            k_start = Lk - WINDOW
        wk = k_end - k_start + 1

        attn_block = attn_head0[q_start:q_end+1, k_start:k_end+1]  # [wq, wk]

        # Q 轴 token 文本（精确对应 xt）
        q_ids = step_token_ids[q_start:q_end+1].tolist()
        q_tokens = [tokenizer.decode([tid]) for tid in q_ids]

        # K 轴：只标 index，避免误导（历史+xt 混合）
        x_pos = list(range(wk))
        y_pos = list(range(wq))
        y_labels = [f"{q_start + i}:{q_tokens[i]}" for i in range(wq)]

        plt.figure(figsize=(12, 8))
        plt.imshow(attn_block, cmap="viridis", aspect="auto")
        plt.colorbar()

        plt.title(
            f"Layer {layer_id} - Step {step_id}\n"
            f"Q[{q_start}~{q_end}] (xt) vs K[{k_start}~{k_end}]"
        )

        plt.yticks(y_pos, y_labels, fontsize=6)
        plt.ylabel("Query = current noisy block xt (index:token)")

        plt.tight_layout()
        plt.savefig(f"{layer_dir}/step_{step_id:03d}.png", dpi=140)
        plt.close()

print("PNG files saved.")

# ===========================================================
# Generate GIFs：按 L_k 变化分 block（对应不同 xt block）
# ===========================================================

print("\nGenerating GIFs per layer (split by L_k changes)...")

for layer_id in save_layers:
    layer_dir = os.path.join(save_dir, f"layer_{layer_id:02d}")
    if not os.path.isdir(layer_dir):
        continue

    png_files = sorted(
        f for f in os.listdir(layer_dir)
        if f.startswith("step_") and f.endswith(".png")
    )
    if not png_files:
        print(f"Layer {layer_id}: no PNG, skip.")
        continue

    prev_Lk = None
    block_id = 0
    frames = []

    trace = layer_traces[layer_id]

    for png in png_files:
        step_id = int(png.split("_")[1].split(".")[0])
        attn = trace[step_id][0, 0]
        Lk = attn.shape[1]

        # 当 L_k 发生变化，认为进入了下一个 xt block
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

# ===========================================================
# Print model output
# ===========================================================

print("\n================ MODEL OUTPUT ================\n")
print(tokenizer.decode(generated_ids[inputs.shape[1]:], skip_special_tokens=True))
print("\n==============================================\n")