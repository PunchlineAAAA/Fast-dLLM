import os
os.environ["PYTORCH_USE_FLASH_ATTENTION"] = "0"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"

import torch
import math
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import types
import generation_functions


# ======================
# 配置
# ======================
model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"

# 选择 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======================
# 加载 tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# ======================
# 加载模型（到 GPU！）
# ======================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.to(device)
model.eval()
print("Model device:", next(model.parameters()).device)


# 绑定扩散采样函数（和 eval.py 一致）
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
    model
)


# ======================
# Hook：抓取最后一层 Q/K
# ======================
diffusion_q = []
diffusion_k = []


def hook_q(module, inputs, output):
    diffusion_q.append(output.detach().float().cpu())
    print("Captured Q:", diffusion_q[-1].shape)

def hook_k(module, inputs, output):
    diffusion_k.append(output.detach().float().cpu())
    print("Captured K:", diffusion_k[-1].shape)


for name, module in model.named_modules():
    if "layers.27.self_attn.q_proj" in name:
        print("Hook on:", name)
        module.register_forward_hook(hook_q)

    if "layers.27.self_attn.k_proj" in name:
        print("Hook on:", name)
        module.register_forward_hook(hook_k)



# ======================
# Prompt 输入
# ======================
prompt = "Josh decides to try flipping a house. He buys a house for $80,000 then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"

inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


# ======================
# 执行一次扩散采样（加入 eval.py 的计速逻辑）
# ======================
print("\nRunning diffusion decoding...\n")
print("Input token length:", inputs.shape[1], "\n")

# === eval.py 标准计时：用 perf_counter，精度最高 ===
start_time = time.perf_counter()

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

end_time = time.perf_counter()
elapsed = end_time - start_time   # eval.py 的写法

print("\nDiffusion decoding finished!")
print(f"Elapsed time (eval.py style): {elapsed:.4f} sec")


# ======================
# eval.py 的 tokens/s 计算方式
# ======================
generated_ids = generated[0]
input_len = inputs.shape[1]
new_tokens = generated_ids.shape[0] - input_len

tps = new_tokens / elapsed
print(f"Generated tokens: {new_tokens}")
print(f"Speed (tokens/s): {tps:.2f}")


# ======================
# 显示 Q/K 信息
# ======================
print("Total diffusion steps captured:", len(diffusion_q))

if len(diffusion_q) > 0 and len(diffusion_k) > 0:
    print("Q.shape:", diffusion_q[0].shape)
    print("K.shape:", diffusion_k[0].shape)
else:
    print("Hooks did NOT capture anything!")


# ======================
# GQA Attention 计算
# ======================
def compute_attention(q, k, num_q_heads=28):
    B, L, Dq = q.shape
    _, _, Dk = k.shape

    head_dim = 128
    num_k_heads = Dk // head_dim  # 4

    q = q.view(B, L, num_q_heads, head_dim).permute(0,2,1,3)
    k = k.view(B, L, num_k_heads, head_dim).permute(0,2,1,3)

    group_size = num_q_heads // num_k_heads
    q_grouped = q.view(B, num_k_heads, group_size, L, head_dim).mean(dim=2)

    scores = torch.matmul(q_grouped, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)

    return attn[0]


# ======================
# 输出注意力图/GIF
# ======================
os.makedirs("diffusion_attn", exist_ok=True)
frames = []

for step, (q, k) in enumerate(zip(diffusion_q, diffusion_k)):
    attn = compute_attention(q, k)

    plt.figure(figsize=(6,5))
    plt.imshow(attn[0], cmap="viridis")
    plt.colorbar()
    plt.title(f"Diffusion Step {step} - Attention Head 0")
    fname = f"diffusion_attn/attn_step_{step}.png"
    plt.savefig(fname, dpi=200)
    plt.close()

    frames.append(imageio.imread(fname))

imageio.mimsave("diffusion_attention.gif", frames, duration=0.2)
print("\nSaved attention maps and GIF.\n")


# ======================
# 打印最终模型输出
# ======================
full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
new_tokens_output = tokenizer.decode(generated_ids[input_len:], skip_special_tokens=True)

print("\n================ MODEL FULL OUTPUT ================\n")
print(full_output)

print("\n================ MODEL GENERATED ANSWER ================\n")
print(new_tokens_output)
print("\n=====================================================\n")
