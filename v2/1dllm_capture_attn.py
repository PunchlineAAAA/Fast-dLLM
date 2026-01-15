import os
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import types
import csv
import generation_functions

from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm

# ===========================================================
# 基本配置
# ===========================================================

MODEL_PATH = "Efficient-Large-Model/Fast_dLLM_v2_7B"

PROMPT = (
    "Josh decides to flip a house. He buys a house for $80,000 and then puts in "
    "$50,000 in repairs. This increased the value of the house by 150%. "
    "How much profit did he make?"
)

BLOCK_SIZE = 32
SMALL_BLOCK_SIZE = 8
MAX_NEW_TOKENS = 2048

MASK_ID = 151665
STOP_TOKEN = 151645
THRESHOLD = 1.0

SAVE_DIR = "diffusion_attn"
SAVE_LAYERS = {5,6,7,23,24}
# SAVE_LAYERS = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}
PLOT_MODE = "both"          # "avg" | "max" | "both"

# ===== NEW: 语义级可视化参数（默认开启，不影响其他功能）=====
IGNORE_DOMINANT_FOR_SCALE = True     # 是否忽略极值列来定义标尺
SCALE_PERCENTILE = 99.5              # 用剩余 attention 的高分位定义 vmax

os.environ["PYTORCH_USE_FLASH_ATTENTION"] = "0"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"
os.environ.setdefault("FLEX_ATTENTION_DISABLE", "0")

# ===========================================================
# 工具函数
# ===========================================================

def clear_all_attn_traces(model):
    for layer in model.model.layers:
        sa = layer.self_attn
        if hasattr(sa, "_attn_trace"):
            sa._attn_trace.clear()
        if hasattr(sa, "_attn_tokens"):
            sa._attn_tokens.clear()
        if hasattr(sa, "_last_attn"):
            sa._last_attn = None


def build_white_to_viridis():
    base = plt.cm.viridis(np.linspace(0, 1, 256))
    white = np.array([[1, 1, 1, 1]])
    return LinearSegmentedColormap.from_list(
        "white_to_viridis",
        np.vstack([white, base])
    )

# ===== NEW: 找全局 dominant key 列 =====
def find_dominant_column(heatmap):
    # heatmap: [step, key]
    col_score = np.nanmean(heatmap, axis=0)
    return int(np.nanargmax(col_score))

# ===== NEW: 构造“语义 LogNorm”（不修改数据本身）=====
def build_semantic_lognorm(
    heatmap,
    ignore_column=None,
    percentile=99.5,
    eps=1e-6,
):
    H = heatmap.copy()

    if ignore_column is not None:
        H[:, ignore_column] = np.nan

    vals = H[np.isfinite(H)]
    vals = vals[vals > 0]

    if vals.size == 0:
        raise RuntimeError("No valid attention values for normalization")

    vmin = max(vals.min(), eps)
    vmax = np.percentile(vals, percentile)

    return LogNorm(vmin=vmin, vmax=vmax)


# ===== MOD: plot_heatmap 支持可选 custom_norm（默认不影响原行为）=====
def plot_heatmap(
    heatmap,
    title,
    out_path,
    use_log=True,
    custom_norm=None,
):
    heatmap = heatmap.copy()

    # attention 为 0 的地方视为“无关注”
    heatmap[heatmap == 0] = np.nan

    eps = 1e-6
    heatmap_plot = heatmap.copy()
    heatmap_plot[np.isnan(heatmap_plot)] = eps
    heatmap_plot[heatmap_plot < eps] = eps

    if custom_norm is not None:
        norm = custom_norm
        cbar_label = "Attention (log scale, semantic)"
    else:
        if use_log:
            vmax = np.nanmax(heatmap_plot)
            norm = LogNorm(vmin=eps, vmax=vmax)
            cbar_label = "Attention (log scale)"
        else:
            vmin = np.nanmin(heatmap_plot)
            vmax = np.nanmax(heatmap_plot)
            norm = Normalize(vmin=vmin, vmax=vmax)
            cbar_label = "Attention (linear)"

    cmap = build_white_to_viridis()

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        heatmap_plot,
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(im, label=cbar_label)
    plt.xlabel("Key token index")
    plt.ylabel("Diffusion step")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ===== 热力图数值分布（尤其是0）诊断 =====
def diagnose_structural_zeros(attn_trace, eps=1e-12, name="heatmap"):
    """
    attn_trace: 原始 sa._attn_trace 列表
                每个元素 shape = [1, H, L_q, L_k_step]
    """
    zero_cnt = 0
    pos_cnt = 0

    for s, attn in enumerate(attn_trace):
        # attn: [1, H, L_q, L_k]
        A = attn.cpu().numpy()

        # collapse to [L_k]，与 heatmap_avg 对齐
        A_mean = A.mean(axis=(0, 1, 2))

        zero_cnt += np.sum(A_mean == 0)
        pos_cnt  += np.sum(A_mean > 0)

    print(
        f"[{name} structural diagnostics]\n"
        f"  structural == 0 : {zero_cnt}\n"
        f"  structural > 0  : {pos_cnt}\n"
        f"  total structural: {zero_cnt + pos_cnt}"
    )

# ===== 找最小 非 0、结构性 attention =====
def find_min_structural_attention(attn_trace):
    """
    在原始 attn_trace 中查找：
    - attention > 0
    - 仅限真实 key 区域（左侧阶梯）
    
    返回：
      (min_value, step, key_idx)
    """
    min_val = None
    min_step = None
    min_key = None

    for s, attn in enumerate(attn_trace):
        # attn: [1, H, L_q, L_k]
        A = attn.cpu().numpy()
        A_mean = A.mean(axis=(0, 1, 2))  # [L_k]

        # 只看 > 0 的
        mask = A_mean > 0
        if not np.any(mask):
            continue

        local_min = A_mean[mask].min()
        if min_val is None or local_min < min_val:
            k = np.where(A_mean == local_min)[0][0]
            min_val = local_min
            min_step = s
            min_key = k

    return min_val, min_step, min_key

def load_token_map(csv_path):
    idx_to_token = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_token[int(row["idx0"])] = row["token_str"]
    return idx_to_token

# ===== 找heatmap_avg中全局最小的 5 个非零值 =====
def print_global_min_k_nonzero_with_token(
    heatmap,
    token_map,
    k=5,
    name="heatmap_avg"
):
    """
    heatmap: 2D array [step, key]
    token_map: dict {key_idx -> token_str}
    """
    mask = heatmap > 0
    if not np.any(mask):
        print(f"[{name}] no positive values found")
        return

    values = heatmap[mask]
    indices = np.argwhere(mask)  # (step, key)

    order = np.argsort(values)
    topk = order[:k]

    print(f"[{name}] global smallest {k} non-zero values:")
    for rank, idx in enumerate(topk, 1):
        s, key_idx = indices[idx]
        v = values[idx]
        tok = token_map.get(key_idx, "<UNK>")

        print(
            f"  #{rank}: "
            f"value={v:.6e}, "
            f"step={s}, "
            f"key_idx={key_idx}, "
            f"token={repr(tok)}"
        )


# ===========================================================
# 主逻辑
# ===========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
        model
    )

    clear_all_attn_traces(model)

    inputs = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

    print("\nRunning diffusion decoding ...")
    generated = model.mdm_sample(
        inputs,
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        small_block_size=SMALL_BLOCK_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        min_len=min(inputs.shape[1], BLOCK_SIZE - 1),
        seq_len=torch.tensor([inputs.shape[1]], device=device),
        mask_id=MASK_ID,
        stop_token=STOP_TOKEN,
        threshold=THRESHOLD,
    )

    generated_ids = generated[0]

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ============================
    # 导出最终 token 序列（从 0 开始编号）
    # ============================

    out_dir = Path(SAVE_DIR)
    
    # 注意：generated_ids 已经是 1D tensor [total_len]
    full_ids = generated_ids.detach().cpu()   # 不要再取 [0]
    
    # 1) 整段文本（sanity check）
    full_text = tokenizer.decode(
        full_ids,
        skip_special_tokens=True
    )
    
    txt_path = out_dir / "generated_full_text.txt"
    txt_path.write_text(full_text, encoding="utf-8")
    
    # 2) CSV：逐 token + 序号
    csv_path = out_dir / "generated_tokens_full.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx0", "token_id", "token_str"])
    
        for i, tid in enumerate(full_ids.tolist()):
            tok = tokenizer.decode([tid])  # 单 token decode，保留空格
            writer.writerow([i, tid, tok])
    
    print(f"[Saved] {txt_path}")
    print(f"[Saved] {csv_path}")

    token_map = load_token_map(
        Path(SAVE_DIR) / "generated_tokens_full.csv"
    )

    for layer_id in SAVE_LAYERS:
        sa = model.model.layers[layer_id].self_attn
        trace = getattr(sa, "_attn_trace", [])

        if not trace:
            print(f"[WARN] Layer {layer_id}: no attention trace")
            continue

        min_val, min_step, min_key = find_min_structural_attention(trace)

        diagnose_structural_zeros(
            trace,
            name=f"Layer {layer_id}"
        )

        max_Lk = max(attn.shape[-1] for attn in trace)
        padded = [
            torch.nn.functional.pad(attn, (0, max_Lk - attn.shape[-1]))
            if attn.shape[-1] < max_Lk else attn
            for attn in trace
        ]

        # [step, 1, H, L_q, L_k]
        attn_tensor = torch.stack(padded, dim=0).cpu()

        # ===== 原始功能：保存 pt（不变）=====
        torch.save(
            attn_tensor,
            Path(SAVE_DIR) / f"layer_{layer_id:02d}_attn_tensor.pt"
        )

        # ===== 原始 heatmap 计算（不变）=====
        heatmap_avg = attn_tensor.mean(dim=(1, 2, 3)).numpy()
        heatmap_max = attn_tensor.max(dim=3).values.mean(dim=(1, 2)).numpy()

        # ===== NEW: 只用于标尺的 dominant key =====
        dominant_col = find_dominant_column(heatmap_avg)

        if IGNORE_DOMINANT_FOR_SCALE:
            col_score = np.nanmean(heatmap_avg, axis=0)
            dominant_value = col_score[dominant_col]
            other = np.delete(col_score, dominant_col)
        
            print(
                f"[Layer {layer_id}] "
                f"IGNORE_DOMINANT_FOR_SCALE=True | "
                f"dominant key column = {dominant_col}, "
                f"mean attention = {dominant_value:.4e}\n"
                f"  other columns: "
                f"mean = {other.mean():.4e}, "
                f"std = {other.std():.4e}, "
                f"max = {other.max():.4e}, "
                f"min = {other.min():.4e}"
            )

        # 输出最小值及对应token
        if min_val is not None:
            tok = token_map.get(min_key, "<UNK>")
            print(
                f"[Layer {layer_id} min attention]\n"
                f"  value   : {min_val:.6e}\n"
                f"  step    : {min_step}\n"
                f"  key_idx : {min_key}\n"
                f"  token   : {repr(tok)}"
            )
        else:
            print(f"[Layer {layer_id}] no positive attention found")

        # 输出heatmap_avg中全局最小的 5 个非零值
        print_global_min_k_nonzero_with_token(
            heatmap_avg,
            token_map,
            k=15,
            name=f"Layer {layer_id} heatmap_avg"
        )


        # ===== AVG =====
        if PLOT_MODE in ("avg", "both"):
            norm = None
            if IGNORE_DOMINANT_FOR_SCALE:
                norm = build_semantic_lognorm(
                    heatmap_avg,
                    ignore_column=dominant_col,
                    percentile=SCALE_PERCENTILE,
                )

            print(
                f"[Layer {layer_id} scale check]\n"
                f"  min_val (raw)      = {min_val:.6e}\n"
                f"  semantic vmin      = {norm.vmin:.6e}\n"
                f"  ratio min/vmin     = {min_val / norm.vmin:.3f}"
            )

            plot_heatmap(
                heatmap_avg,
                title=f"Diffusion Attention (AVG over Q) - Layer {layer_id}",
                out_path=Path(SAVE_DIR) / f"layer_{layer_id:02d}_heatmap_avg.png",
                use_log=True,
                custom_norm=norm,
            )

        # ===== MAX =====
        if PLOT_MODE in ("max", "both"):
            norm = None
            if IGNORE_DOMINANT_FOR_SCALE:
                norm = build_semantic_lognorm(
                    heatmap_max,
                    ignore_column=dominant_col,
                    percentile=SCALE_PERCENTILE,
                )

            plot_heatmap(
                heatmap_max,
                title=f"Diffusion Attention (MAX over Q) - Layer {layer_id}",
                out_path=Path(SAVE_DIR) / f"layer_{layer_id:02d}_heatmap_max.png",
                use_log=True,
                custom_norm=norm,
            )

        print(f"Saved AVG/MAX heatmaps for layer {layer_id}")

    # =======================================================
    # 打印最终输出（sanity check）
    # =======================================================

    print("\n================ MODEL OUTPUT ================\n")
    print(tokenizer.decode(
        generated_ids[inputs.shape[1]:],
        skip_special_tokens=True
    ))
    print("\n=============================================\n")
    
    # print("\nDone.")

if __name__ == "__main__":
    main()