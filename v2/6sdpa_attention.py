import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 1. 加载 Fast-dLLM v2
# =========================
model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)
model.eval()
print(">>> Model loaded on", device)

# =========================
# 2. 拿到 sdpa_attention_forward
# =========================
attn = model.model.layers[0].self_attn
glob = attn.forward.__globals__

all_funcs = glob["ALL_ATTENTION_FUNCTIONS"]
orig_sdpa = all_funcs["sdpa"]
print(">>> Found sdpa_attention_forward:", orig_sdpa)

captured = []


# =========================
# 3. 包一层 sdpa：保持原行为 + 额外计算 attn_probs
# =========================
def sdpa_with_attn(*args, **kwargs):
    # Run original sdpa (for context)
    context, _ = orig_sdpa(*args, **kwargs)

    self_mod = args[0]
    q = args[1]   # [B, 28, L, D]
    k = args[2]   # [B, 4,  L, D]

    # ==== 1. GQA：把 28 Q-head 分组成 4 组（KV heads） ====
    B, Hq, L, D = q.shape
    _, Hk, _, _ = k.shape

    group = Hq // Hk   # 28 / 4 = 7
    q = q.view(B, Hk, group, L, D).mean(dim=2)   # → [B, 4, L, D]

    # ==== 2. attention mask ====
    attn_mask = None
    if len(args) >= 5:
        attn_mask = args[4]
    if attn_mask is None:
        attn_mask = kwargs.get("attention_mask", None)

    # ==== 3. scaling ====
    scaling = kwargs.get("scaling", getattr(self_mod, "scaling", 1.0))

    # ==== 4. compute scores ====
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling

    if attn_mask is not None:
        scores = scores + attn_mask

    attn_probs = torch.softmax(scores, dim=-1)

    if not captured:
        captured.append(attn_probs.detach().cpu())
        print(">>> Captured real attn_probs:", attn_probs.shape)

    return context, attn_probs


# 注册 patch
all_funcs["sdpa"] = sdpa_with_attn
print(">>> Patched ALL_ATTENTION_FUNCTIONS['sdpa'] with sdpa_with_attn")


# =========================
# 4. 跑一次普通 forward
# =========================
prompt = "Hello Fast-dLLM attention hooking."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(">>> Running forward...")
with torch.no_grad():
    _ = model(**inputs)

print(">>> Forward done.")
print("Number of captured attention tensors:", len(captured))
if captured:
    print("Example captured attn_probs shape:", captured[0].shape)
