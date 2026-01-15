import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

# =======================
# 配置
# =======================

ATTN_PT = "diffusion_attn/layer_24_attn_tensor.pt"
TOKEN_CSV = "diffusion_attn/generated_tokens_full.csv"

TOP_K = 5        # 画 attention 最高的 K 个 token
SKIP_TOP = 2       # 跳过最强 anchor（例如 'for'）

SAVE_FIG = "diffusion_attn/top_attended_token_curves.png"

# =======================
# 加载 attention
# =======================

print("Loading attention tensor...")
attn = torch.load(ATTN_PT)   # [step, 1, H, L_q, L_k]
print("Shape:", attn.shape)

# =======================
# 加载 token 映射（idx0 -> token_str）
# =======================

idx_to_token = {}

with open(TOKEN_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row["idx0"])
        tok = row["token_str"]
        idx_to_token[idx] = tok

print(f"Loaded {len(idx_to_token)} tokens from CSV.")

def key_to_str(k):
    # 如果越界，给一个兜底
    return idx_to_token.get(k, "<out-of-range>")

# =======================
# 计算 attention 聚合
# =======================

# mean over batch & head
# -> [step, L_q, L_k]
attn_qk = attn.mean(dim=(1, 2))

# avg / max over Q
# -> [step, L_k]
attn_avg = attn_qk.mean(dim=1)
attn_max = attn_qk.max(dim=1).values

# total attention per key
attn_key_mass = attn_avg.sum(dim=0)

# =======================
# 自动选 token
# =======================

top_vals, top_idx = torch.topk(attn_key_mass, TOP_K + SKIP_TOP)
selected_idx = top_idx[SKIP_TOP:]

print("\nSelected key indices:")
for i, idx in enumerate(selected_idx.tolist()):
    print(f"{i:2d}. key_index = {idx}, token = '{key_to_str(idx)}'")

# =======================
# 画折线图
# =======================

steps = np.arange(attn_avg.shape[0])

plt.figure(figsize=(11, 6))

for k in selected_idx.tolist():
    tok = key_to_str(k)
    label_base = f"k={k} ('{tok}')"

    plt.plot(
        steps,
        attn_avg[:, k].numpy(),
        label=f"{label_base} avg",
        linewidth=2
    )
    plt.plot(
        steps,
        attn_max[:, k].numpy(),
        linestyle="--",
        alpha=0.8,
        label=f"{label_base} max"
    )

plt.xlabel("Diffusion step")
plt.ylabel("Attention strength")
plt.title("Top-attended tokens over diffusion steps (Layer 24)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(SAVE_FIG, dpi=150)
plt.close()

print(f"\nSaved figure to {SAVE_FIG}")