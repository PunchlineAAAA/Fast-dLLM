import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib.colors import LogNorm

# =======================
# 配置
# =======================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = (
    "Josh decides to flip a house. He buys a house for $80,000 and then puts in "
    "$50,000 in repairs. This increased the value of the house by 150%. "
    "How much profit did he make?"
)
MAX_NEW_TOKENS = 1024
LAYER_IDX = 15
SAVE_DIR = Path("qwen_attn")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ATTN_PT = SAVE_DIR / "layer_attn_tensor.pt"
TOKEN_CSV = SAVE_DIR / "generated_tokens_full.csv"

# =======================
# Load model
# =======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device)
model.eval()

# =======================
# Generate with attentions
# =======================

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        return_dict_in_generate=True,
        output_attentions=True,
    )

attentions = out.attentions
sequences = out.sequences[0].detach().cpu()   # [total_len]

prompt_len = inputs["input_ids"].shape[1]
total_len = sequences.shape[0]

print(f"Prompt len: {prompt_len}, total len: {total_len}")
print(f"Captured {len(attentions)} generation steps")

# =======================
# 保存 token CSV（0-based）
# =======================

with open(TOKEN_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["idx0", "token_id", "token_str"])
    for i, tid in enumerate(sequences.tolist()):
        tok = tokenizer.decode([tid])
        writer.writerow([i, tid, tok])

print(f"[Saved] {TOKEN_CSV}")

# =======================
# 构建 attention tensor（仅保留“新 token”的 Q）
# =======================

# attentions: list[step] -> tuple[layer] -> [1, H, Q, K]
num_steps = len(attentions)

attn_steps = []

# 找最大 K（历史长度）
max_K = max(
    attentions[s][LAYER_IDX].shape[-1]
    for s in range(num_steps)
)

for step in range(num_steps):
    layer_attn = attentions[step][LAYER_IDX]  # [1, H, Q, K]

    # ===== 核心修复点 =====
    # 只取“新生成 token”的 attention
    # - prefill(step=0): Q=prompt_len → 取最后一个
    # - decode(step>=1): Q=1 → 本身就是最后一个
    layer_attn = layer_attn[:, :, -1:, :]     # [1, H, 1, K]

    K = layer_attn.shape[-1]
    if K < max_K:
        pad = max_K - K
        layer_attn = torch.nn.functional.pad(layer_attn, (0, pad))

    attn_steps.append(layer_attn)

# [step, 1, H, 1, K]
attn_tensor = torch.stack(attn_steps, dim=0).cpu()
torch.save(attn_tensor, ATTN_PT)

print(f"[Saved] attention tensor: {ATTN_PT}")
print("Final attention shape:", attn_tensor.shape)

# =======================
# 画 attention 热力图（AR step × Key）
# =======================

# attn_tensor: [step, 1, H, 1, K]
# → [step, K]
heatmap = attn_tensor.mean(dim=(1, 2, 3)).float().numpy()

# 把 0 当作 padding → NaN
heatmap[heatmap == 0] = np.nan

# ===== LogNorm 需要正数 =====
eps = 1e-6
heatmap_plot = heatmap.copy()

# NaN / padding → eps（否则 LogNorm 会炸）
heatmap_plot[np.isnan(heatmap_plot)] = eps
heatmap_plot[heatmap_plot < eps] = eps

vmax = heatmap_plot.max()
norm = LogNorm(vmin=eps, vmax=vmax)


# 白 → viridis colormap
base = plt.cm.viridis(np.linspace(0, 1, 256))
white = np.array([[1, 1, 1, 1]])
cmap = LinearSegmentedColormap.from_list(
    "white_to_viridis",
    np.vstack([white, base])
)

# 画图
plt.figure(figsize=(12, 7))
im = plt.imshow(
    heatmap_plot,
    aspect="auto",
    cmap=cmap,
    norm=norm,
)

plt.colorbar(im, label="Attention strength")
plt.xlabel("Key token index (prompt + generated)")
plt.ylabel("AR generation step")
plt.title(f"Qwen2.5-7B-Instruct Attention Heatmap (Layer {LAYER_IDX})")

plt.tight_layout()
out_path = SAVE_DIR / f"layer_{LAYER_IDX:02d}_attn_heatmap.png"
plt.savefig(out_path, dpi=150)
plt.close()

print(f"[Saved] attention heatmap: {out_path}")