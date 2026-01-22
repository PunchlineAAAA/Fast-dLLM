"""Attention entropy probe for block-diffusion LLMs.
本脚本针对Fast-dLLM扩散采样过程记录注意力权重，计算注意力熵，
用于定位“决策性”层/步（attention集中代表模型可能在做决定）。
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import types
import generation_functions

# ----------------------------------------------------------------------
# 禁用加速注意力后端，强制使用标准SDPA以确保能完整捕获注意力权重
# （否则flash-attention等后端可能不返回中间注意力映射）
# ----------------------------------------------------------------------
os.environ.setdefault("PYTORCH_USE_FLASH_ATTENTION", "0")
os.environ.setdefault("PYTORCH_ENABLE_SDPA", "0")
os.environ.setdefault("FLEX_ATTENTION_DISABLE", "0")

# =========================
# Dataclasses 数据结构定义
# =========================


@dataclass
class StepEntropy:
    # 一个生成步骤(step)内单层(layer)的注意力统计
    layer: int  # 记录是模型的第几层
    step: int  # 采样生成的第几步
    concentration: float  # 注意力集中度指标（1-平均熵/log(K)）
    concentration_corrected: float  # 使用可见 key 数修正后的集中度
    head_entropy_mean: List[float]  # 各注意力头熵平均值
    head_entropy_quantiles: Dict[str, float]  # 熵分位数(衡量是否集中某些头)
    head_variance: float  # 注意力头间熵的方差(偏好差异)
    seq_len_q: int  # Q序列长度（查询长度）
    seq_len_k: int  # K序列长度（键长度）
    effective_key_count: float  # 实际可见 key 数（按 mask 修正后的平均）
    visible_key_ratio: float  # 可见 key 比例 (effective_key_count / seq_len_k)
    token_persistence_entropy: float  # token跨步稳定性熵（越低越稳定）
    top_token_jaccard: float  # token关注集合相似度（Jaccard）
    token_stability: float  # 当前 step token 到下一步的稳定性
    top_token_positions: List[int]  # 被关注的 key 位置索引
    top_token_ids: List[int]  # 被关注的 token id
    top_token_strs: List[str]  # 被关注的 token 字符


@dataclass
class PromptReport:
    prompt: str  # 输入提示词
    steps: List[StepEntropy]  # 该prompt的所有统计记录


# =========================
# Attention utilities
# =========================


def clear_all_attn_traces(model: AutoModelForCausalLM) -> None:
    """
    清除模型内部记录的注意力轨迹，避免不同prompt之间干扰。
    """
    for layer in model.model.layers:
        sa = layer.self_attn
        if hasattr(sa, "_attn_trace"):
            sa._attn_trace.clear()
        if hasattr(sa, "_last_attn"):
            sa._last_attn = None
        if hasattr(sa, "_attn_tokens"):
            sa._attn_tokens.clear()
        if hasattr(sa, "_attn_mask"):
            sa._attn_mask.clear()


def attach_sampler(model: AutoModelForCausalLM) -> None:
    """
    将Fast-dLLM扩散采样函数绑定为model.mdm_sample方法，
    方便直接调用并捕获中间attention。
    """
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
        model,
    )


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    """
    计算熵 H(p) = -sum(p log p)
    注意加入 clamp_min 防止log(0)。
    """
    safe_prob = prob.clamp_min(1e-8)
    return -(safe_prob * safe_prob.log()).sum(dim=-1)


def _token_persistence(key_focus_steps: Sequence[torch.Tensor]) -> float:
    """
    评估token关注跨多个step的稳定性。
    若某些token持续被关注，则熵较低 → 决策更集中/路径更加一致。
    """
    if len(key_focus_steps) < 2:
        return 0.0

    stacked = torch.stack(key_focus_steps, dim=0)  # [S, K]
    token_sum = stacked.sum(dim=0)

    if token_sum.sum() == 0:
        return 0.0

    step_distributions = []
    for token_idx in range(stacked.shape[1]):
        token_over_steps = stacked[:, token_idx]
        total = token_over_steps.sum()
        if total == 0:
            continue
        probs = token_over_steps / total
        step_distributions.append(_entropy(probs.unsqueeze(0)))

    if not step_distributions:
        return 0.0
    return torch.stack(step_distributions).mean().item()


def _top_token_jaccard(token_dists: Sequence[torch.Tensor], top_t: int) -> float:
    """
    比较相邻步骤被关注最多的top_k token的一致性，
    使用Jaccard(similarity = |A∩B| / |A∪B|)，价值在于衡量注意力是否持续集中。
    """
    if len(token_dists) < 2:
        return 0.0

    overlaps = []
    for prev, nxt in zip(token_dists[:-1], token_dists[1:]):
        prev_top = torch.topk(prev, k=min(top_t, prev.numel()), dim=-1).indices
        next_top = torch.topk(nxt, k=min(top_t, nxt.numel()), dim=-1).indices
        prev_set, next_set = set(prev_top.tolist()), set(next_top.tolist())
        union = len(prev_set | next_set)
        overlaps.append(0.0 if union == 0 else len(prev_set & next_set) / union)

    return float(mean(overlaps))


def _step_token_stability(token_steps: Sequence[torch.Tensor]) -> List[float]:
    """
    计算每一步 token 是否在下一步发生变化，返回每步稳定率。
    稳定率 = 相同 token 数量 / 可比较 token 数量。
    """
    if not token_steps:
        return []

    stabilities = []
    for idx in range(len(token_steps)):
        if idx >= len(token_steps) - 1:
            stabilities.append(0.0)
            continue
        current = token_steps[idx].squeeze(0)
        nxt = token_steps[idx + 1].squeeze(0)
        min_len = min(current.numel(), nxt.numel())
        if min_len == 0:
            stabilities.append(0.0)
            continue
        stable = (current[:min_len] == nxt[:min_len]).float().mean().item()
        stabilities.append(stable)
    return stabilities


def _effective_key_count(
    attn: torch.Tensor, mask: torch.Tensor | None
) -> float:
    if mask is None:
        visible_mask = attn > 1e-8
        return visible_mask.float().sum(dim=-1).mean().item()

    mask_tensor = mask
    if mask_tensor.dtype != torch.bool:
        mask_tensor = mask_tensor > 0

    if mask_tensor.dim() == 4:
        mask_tensor = mask_tensor[:, 0]
    elif mask_tensor.dim() == 3:
        pass
    elif mask_tensor.dim() == 2:
        mask_tensor = mask_tensor[:, None, :]
    else:
        visible_mask = attn > 1e-8
        return visible_mask.float().sum(dim=-1).mean().item()

    return mask_tensor.float().sum(dim=-1).mean().item()


# =========================
# Core metrics
# =========================


def compute_layer_metrics(
    trace: Sequence[torch.Tensor],
    token_steps: Sequence[torch.Tensor],
    mask_steps: Sequence[torch.Tensor],
    block_size: int,
    top_t: int,
    tokenizer: AutoTokenizer,
) -> List[StepEntropy]:
    """
    对某一层(layer)记录的attention轨迹trace按step聚合，
    计算每个step的注意力熵与集中度指标。
    """
    stats = []
    if not trace:
        return stats

    key_focus_steps = []
    token_top_sets = []
    token_stability_steps = _step_token_stability(token_steps)

    for step_idx, attn in enumerate(trace):
        # attn形状: [B, H, Q, K] = batch, head数, 查询长度, 键长度
        batch_size, num_heads, q_len, k_len = attn.shape

        # 计算熵
        ent = _entropy(attn)  # [B, H, Q]
        head_mean = ent.mean(dim=(0, 2))  # 按head平均 -> 每个头的平均熵
        layer_mean = ent.mean()

        # 注意力集中度: 1 - (平均熵/log(K))
        # 越接近1代表注意力越集中，越接近0说明越均匀(模型不确定)
        concentration = 1.0 - layer_mean.item() / math.log(k_len)

        # 可见 key 数修正，避免 mask 导致的虚高集中度
        mask = mask_steps[step_idx] if step_idx < len(mask_steps) else None
        effective_k = _effective_key_count(attn, mask)
        safe_effective_k = max(effective_k, 2.0)
        concentration_corrected = 1.0 - layer_mean.item() / math.log(safe_effective_k)
        visible_key_ratio = effective_k / k_len if k_len > 0 else 0.0

        # 按分位数提供每层注意力分布差异
        head_flat = head_mean.flatten()
        quantiles = {
            "p10": torch.quantile(head_flat, 0.10).item(),
            "p50": torch.quantile(head_flat, 0.50).item(),
            "p90": torch.quantile(head_flat, 0.90).item(),
        }
        head_var = head_mean.var(unbiased=False).item()

        # 计算对block尾部token的关注（对应局部Block-Diffusion）
        key_focus = attn.mean(dim=(0, 1, 2))  # [K] 取所有(Q,H,B)平均
        if k_len >= block_size:
            key_focus_tail = key_focus[-block_size:]
        else:
            pad = torch.zeros(block_size - k_len, device=key_focus.device)
            key_focus_tail = torch.cat([pad, key_focus], dim=0)

        # 标准化，用于跨步比较
        normalized_tail = key_focus_tail / (key_focus_tail.sum() + 1e-8)
        key_focus_steps.append(normalized_tail)
        token_top_sets.append(normalized_tail)

        top_positions = (
            torch.topk(key_focus, k=min(top_t, key_focus.numel()), dim=-1).indices
        )
        top_positions_list = top_positions.tolist()
        top_token_ids: List[int] = []
        top_token_strs: List[str] = []
        if step_idx < len(token_steps):
            step_tokens = token_steps[step_idx].squeeze(0)
            if step_tokens.numel() >= k_len:
                key_tokens = step_tokens[-k_len:]
            else:
                key_tokens = step_tokens
            for pos in top_positions_list:
                if pos < key_tokens.numel():
                    token_id = int(key_tokens[pos])
                    top_token_ids.append(token_id)
            if top_token_ids:
                top_token_strs = tokenizer.convert_ids_to_tokens(top_token_ids)

        stats.append(
            StepEntropy(
                layer=-1,  # 稍后填充
                step=step_idx,
                concentration=concentration,
                concentration_corrected=concentration_corrected,
                head_entropy_mean=head_mean.tolist(),
                head_entropy_quantiles=quantiles,
                head_variance=head_var,
                seq_len_q=q_len,
                seq_len_k=k_len,
                effective_key_count=effective_k,
                visible_key_ratio=visible_key_ratio,
                token_persistence_entropy=0.0,  # 下一步赋值
                top_token_jaccard=0.0,  # 下一步赋值
                token_stability=(
                    token_stability_steps[step_idx]
                    if step_idx < len(token_stability_steps)
                    else 0.0
                ),
                top_token_positions=top_positions_list,
                top_token_ids=top_token_ids,
                top_token_strs=top_token_strs,
            )
        )

    # 跨step全局指标
    persistence = _token_persistence(key_focus_steps)
    overlap = _top_token_jaccard(token_top_sets, top_t=top_t)

    for item in stats:
        item.token_persistence_entropy = persistence
        item.top_token_jaccard = overlap

    return stats


def summarize_cross_prompt(
    reports: Sequence[PromptReport],
) -> Dict[int, Dict[str, float]]:
    """
    聚合所有prompt的注意力集中度，生成跨prompt的layer稳定趋势统计。
    """
    bucket = {}
    corrected_bucket = {}
    visible_ratio_bucket = {}
    for report in reports:
        for step in report.steps:
            bucket.setdefault(step.layer, []).append(step.concentration)
            corrected_bucket.setdefault(step.layer, []).append(
                step.concentration_corrected
            )
            visible_ratio_bucket.setdefault(step.layer, []).append(step.visible_key_ratio)

    summary = {}
    for layer_id, concentrations in bucket.items():
        summary[layer_id] = {
            "mean_concentration": mean(concentrations),
            "std_concentration": (
                pstdev(concentrations) if len(concentrations) > 1 else 0.0
            ),
            "mean_concentration_corrected": mean(
                corrected_bucket.get(layer_id, [])
            ),
            "mean_visible_key_ratio": mean(visible_ratio_bucket.get(layer_id, [])),
        }
    return summary


# =========================
# Probe runner 主流程
# =========================


def run_probe(
    model_name: str,
    prompts: Sequence[str],
    block_size: int,
    small_block_size: int,
    top_t: int,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[List[PromptReport], Dict[int, Dict[str, float]]]:
    """
    加载模型 → 绑定采样器 → 执行扩散采样 → 抓取注意力轨迹 → 计算指标。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # AutoModel加载Fast-dLLM模型, 用bfloat16以节省显存
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    attach_sampler(model)

    reports = []

    for prompt in tqdm(prompts, desc="Probing prompts", dynamic_ncols=True):
        clear_all_attn_traces(model)

        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        seq_len = inputs.shape[1]

        # ---------------------------------------------------------------
        # 执行 Fast-dLLM 扩散采样
        # 注意：设置 block_size / small_block_size / threshold
        # 直接影响扩散步数和注意力pattern，需要和训练一致
        # ---------------------------------------------------------------
        _ = model.mdm_sample(
            inputs,
            tokenizer=tokenizer,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
            small_block_size=small_block_size,
            min_len=min(seq_len, block_size - 1),
            seq_len=torch.tensor([seq_len], device=device),
            mask_id=151665,  # Qwen系MASK
            stop_token=151645,  # Qwen系终止符
            threshold=0.9,  # 动态阈值 early-stop/entropy控制
        )

        prompt_steps = []
        for layer_id, layer in enumerate(model.model.layers):
            sa = layer.self_attn
            trace = getattr(sa, "_attn_trace", [])
            token_steps = getattr(sa, "_attn_tokens", [])
            mask_steps = getattr(sa, "_attn_mask", [])
            layer_stats = compute_layer_metrics(
                trace, token_steps, mask_steps, block_size, top_t, tokenizer
            )
            for stat in layer_stats:
                stat.layer = layer_id
                prompt_steps.append(stat)

        reports.append(PromptReport(prompt=prompt, steps=prompt_steps))

    summary = summarize_cross_prompt(reports)
    return reports, summary


# =========================
# Main CLI入口
# =========================


def main(argv: Iterable[str] | None = None) -> None:
    """
    命令行入口，可用 --jsonl-prompts-file / --prompts 覆盖输入。
    运行完成后输出JSON报告供分析。
    """
    parser = argparse.ArgumentParser(
        description="Attention entropy probe for Fast-dLLM diffusion models."
    )
    parser.add_argument(
        "--model", type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B"
    )
    parser.add_argument("--prompts", type=str, nargs="*")
    parser.add_argument("--prompts-file", type=str)
    parser.add_argument(
        "--jsonl-prompts-file",
        type=str,
        help="JSONL file with one object per line containing a 'question' field.",
    )
    parser.add_argument("--output", type=str, default="entropy_report.json")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--top-t", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args(list(argv) if argv is not None else None)

    # -------- default prompt pool --------
    prompt_pool: List[str] = [
        "Josh decides to flip a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    ]

    if args.prompts:
        prompt_pool = args.prompts

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompt_pool = [line.strip() for line in f if line.strip()]

    if args.jsonl_prompts_file:
        prompt_pool = []
        with open(args.jsonl_prompts_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_no}") from e
                if "question" not in obj:
                    raise KeyError(f"Missing 'question' at line {line_no}")
                prompt_pool.append(obj["question"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reports, summary = run_probe(
        model_name=args.model,
        prompts=prompt_pool,
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        top_t=args.top_t,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    payload = {
        "prompts": [asdict(r) for r in reports],
        "cross_prompt_summary": summary,
        "metadata": {
            "model": args.model,
            "device": str(device),
            "block_size": args.block_size,
            "small_block_size": args.small_block_size,
            "top_t": args.top_t,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote entropy report to {args.output}")


if __name__ == "__main__":
    main()
