"""
Attention entropy probe for Qwen2.5-7B (standard causal LM).
Supports multi-prompt probing and outputs JSON in the same structure
as the Fast-dLLM reference script.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================================
# 数据结构
# ======================================================

@dataclass
class StepEntropy:
    layer: int
    step: int
    concentration: float
    head_entropy_mean: List[float]
    head_entropy_quantiles: Dict[str, float]
    head_variance: float
    seq_len_q: int
    seq_len_k: int


@dataclass
class PromptReport:
    prompt: str
    steps: List[StepEntropy]


# ======================================================
# 指标计算
# ======================================================

def _entropy(prob: torch.Tensor) -> torch.Tensor:
    safe_prob = prob.clamp_min(1e-8)
    return -(safe_prob * safe_prob.log()).sum(dim=-1)


def _layer_metrics(attn: torch.Tensor) -> Dict[str, object]:
    # attn: [B,H,Q,K]
    if attn.dtype not in (torch.float32, torch.float64):
        attn = attn.float()

    _, _, q_len, k_len = attn.shape
    ent = _entropy(attn)                      # [B,H,Q]
    head_mean = ent.mean(dim=(0, 2))          # [H]
    layer_mean = ent.mean()                   # scalar

    # 归一化集中度
    concentration = 1.0 - layer_mean.item() / math.log(k_len)

    head_flat = head_mean.flatten()
    quantiles = {
        "p10": torch.quantile(head_flat, 0.10).item(),
        "p50": torch.quantile(head_flat, 0.50).item(),
        "p90": torch.quantile(head_flat, 0.90).item(),
    }
    head_var = head_mean.var(unbiased=False).item()

    return {
        "concentration": concentration,
        "head_entropy_mean": head_mean.tolist(),
        "head_entropy_quantiles": quantiles,
        "head_variance": head_var,
        "seq_len_q": q_len,
        "seq_len_k": k_len,
    }


# ======================================================
# 单prompt运行
# ======================================================

def run_single_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
) -> PromptReport:

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)

    # >>> 修复点：显式构建 position_ids，送到 GPU <<<
    position_ids = torch.arange(
        0, input_ids.shape[1], dtype=torch.long, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,           # <<< 必须加
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("No attentions returned. Use output_attentions=True.")

    steps: List[StepEntropy] = []
    for step_idx, step_attn in enumerate(attentions):
        for layer_idx, layer_attn in enumerate(step_attn):
            if layer_attn is None:
                continue
            m = _layer_metrics(layer_attn)
            steps.append(StepEntropy(
                layer=layer_idx,
                step=step_idx,
                concentration=m["concentration"],
                head_entropy_mean=m["head_entropy_mean"],
                head_entropy_quantiles=m["head_entropy_quantiles"],
                head_variance=m["head_variance"],
                seq_len_q=m["seq_len_q"],
                seq_len_k=m["seq_len_k"],
            ))
    return PromptReport(prompt=prompt, steps=steps)



# ======================================================
# 跨 prompt 汇总（与参考格式一致）
# ======================================================

def summarize_cross_prompt(
    reports: Sequence[PromptReport],
) -> Dict[int, Dict[str, float]]:
    bucket = {}
    for rpt in reports:
        for st in rpt.steps:
            bucket.setdefault(st.layer, []).append(st.concentration)

    summary = {}
    for layer_id, vals in bucket.items():
        summary[layer_id] = {
            "mean_concentration": mean(vals),
            "std_concentration": pstdev(vals) if len(vals) > 1 else 0.0,
        }
    return summary


# ======================================================
# 主入口（与参考脚本接口一致）
# ======================================================

def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Attention entropy probe for Qwen2.5-7B.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompts", type=str, nargs="*")
    parser.add_argument("--prompts-file", type=str)
    parser.add_argument("--jsonl-prompts-file", type=str)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="entropy_report.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # default prompts（与参考一致）
    prompt_pool = [
        "Josh decides to flip a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    ]

    # 覆盖逻辑（与参考一致）
    if args.prompts:
        prompt_pool = args.prompts

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompt_pool = [ln.strip() for ln in f if ln.strip()]

    if args.jsonl_prompts_file:
        prompt_pool = []
        with open(args.jsonl_prompts_file, "r", encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON at line {i}")
                if "question" not in obj:
                    raise KeyError(f"Missing 'question' at line {i}")
                prompt_pool.append(obj["question"])

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    # 修复 generate 警告
    if hasattr(model, "generation_config"):
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.do_sample = (args.temperature > 0)


    # 逐 prompt 运行
    reports = []
    for p in tqdm(prompt_pool, desc="Probing prompts", unit="prompt"):
        rpt = run_single_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            max_new_tokens=args.max_new_tokens,
            device=device,
            temperature=args.temperature,
        )
        reports.append(rpt)


    # 生成与“参考代码完全一致”的 JSON 输出结构
    summary = summarize_cross_prompt(reports)

    payload = {
        "prompts": [asdict(r) for r in reports],
        "cross_prompt_summary": summary,
        "metadata": {
            "model": args.model,
            "device": str(device),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote entropy report to {args.output}")


if __name__ == "__main__":
    main()