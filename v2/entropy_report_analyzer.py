"""
Utility to analyze entropy_report.json produced by attention_entropy_probe.py.

它会对 attention_entropy_probe.py 产生的 entropy_report.json 进行分析，
并按 layer / step 总结以下指标：
- concentration（注意力集中度）
- concentration_corrected（可见 key 修正后的集中度）
- top_token_jaccard（Top Token 集合的 Jaccard 相似度）
- head_variance（多头注意力方差）
- token_stability（token 在相邻 step 的稳定性）

Usage:
    python v2/entropy_report_analyzer.py --report entropy_report.json --top 3
"""

import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def load_report(path: str) -> Dict:
    """加载 JSON 报告文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_steps(steps: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    """
    将单个 prompt 内的 step 信息整理为 {(layer, step): 指标数据}
    每一个 step 是模型采样过程中的一步（扩散/迭代次数等）
    """
    table = {}
    for s in steps:
        # 使用 (layer, step) 作为 key，方便聚合对比
        key = (s["layer"], s["step"])
        table[key] = {
            "concentration": s.get("concentration"),
            "concentration_corrected": s.get("concentration_corrected"),
            "top_token_jaccard": s.get("top_token_jaccard"),
            "head_variance": s.get("head_variance"),
            "effective_key_count": s.get("effective_key_count"),
            "visible_key_ratio": s.get("visible_key_ratio"),
            "token_stability": s.get("token_stability"),
            "token_persistence_entropy": s.get("token_persistence_entropy"),
            "top_token_ids": s.get("top_token_ids"),
            "top_token_strs": s.get("top_token_strs"),
        }
    return table


def aggregate_prompts(
    prompts: List[Dict],
) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
    """
    对多个 prompt 的结果进行汇总，
    聚合每个 (layer, step) 的多个样本值，用 list 保存准备做均值等统计。
    """
    agg: Dict[Tuple[int, int], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for p in prompts:
        table = summarize_steps(p["steps"])
        for key, vals in table.items():
            for metric, v in vals.items():
                if isinstance(v, list):
                    continue
                if v is not None:
                    agg[key][metric].append(v)
    return agg


def mean(xs: List[float]) -> float:
    """计算平均值（避免空列表）"""
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    sorted_xs = sorted(xs)
    pos = (len(sorted_xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] * (hi - pos) + sorted_xs[hi] * (pos - lo)


def format_row(layer: int, step: int, vals: Dict[str, List[float]]) -> str:
    """格式化输出行，展示各类指标的均值和样本数量"""
    conc = vals.get("concentration", [])
    conc_corr = vals.get("concentration_corrected", [])
    jac = vals.get("top_token_jaccard", [])
    var = vals.get("head_variance", [])
    eff_k = vals.get("effective_key_count", [])
    vis_ratio = vals.get("visible_key_ratio", [])
    tok_stab = vals.get("token_stability", [])
    return (
        f"layer={layer:02d} step={step:03d} "
        f"conc_mean={mean(conc):.4f} conc_corr={mean(conc_corr):.4f} "
        f"conc_n={len(conc)} "
        f"jacc_mean={mean(jac):.4f} jacc_n={len(jac)} "
        f"head_var_mean={mean(var):.4f} "
        f"effective_k={mean(eff_k):.1f} visible_ratio={mean(vis_ratio):.3f} "
        f"token_stability={mean(tok_stab):.3f}"
    )


def _build_layer_step_table(
    agg: Dict[Tuple[int, int], Dict[str, List[float]]]
) -> Dict[int, Dict[int, Dict[str, float]]]:
    table: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(dict)
    for (layer, step), vals in agg.items():
        table[layer][step] = {
            "concentration": mean(vals.get("concentration", [])),
            "concentration_corrected": mean(
                vals.get("concentration_corrected", [])
            ),
            "head_variance": mean(vals.get("head_variance", [])),
            "token_stability": mean(vals.get("token_stability", [])),
        }
    return table


def _classify_layers(
    layer_steps: Dict[int, Dict[int, Dict[str, float]]],
    high_percentile: float,
    stable_std: float,
    surge_delta: float,
) -> Dict[int, str]:
    all_conc = [
        step_vals["concentration_corrected"]
        for steps in layer_steps.values()
        for step_vals in steps.values()
    ]
    high_threshold = percentile(all_conc, high_percentile)
    layer_variances = {
        layer: mean([v["head_variance"] for v in steps.values()])
        for layer, steps in layer_steps.items()
        if steps
    }
    variance_threshold = percentile(
        list(layer_variances.values()), high_percentile
    )
    labels: Dict[int, str] = {}
    for layer, steps in layer_steps.items():
        series = [
            steps[step]["concentration_corrected"] for step in sorted(steps)
        ]
        if not series:
            continue
        series_std = std(series)
        layer_mean = mean(series)
        third = max(1, len(series) // 3)
        early_mean = mean(series[:third])
        late_mean = mean(series[-third:])
        if layer_mean >= high_threshold and series_std <= stable_std:
            labels[layer] = "stable_high"
        elif (late_mean - early_mean) >= surge_delta and late_mean >= high_threshold:
            labels[layer] = "late_surge"
        elif layer_variances.get(layer, 0.0) >= variance_threshold:
            labels[layer] = "high_variance"
        else:
            labels[layer] = "modulated"
    return labels


def _layer_token_alignment(
    layer_steps: Dict[int, Dict[int, Dict[str, float]]]
) -> Dict[int, float]:
    corr: Dict[int, float] = {}
    for layer, steps in layer_steps.items():
        sorted_steps = sorted(steps)
        conc = [steps[s]["concentration_corrected"] for s in sorted_steps]
        stability = [steps[s]["token_stability"] for s in sorted_steps]
        if len(conc) < 2 or len(stability) < 2:
            continue
        conc_mean = mean(conc)
        stab_mean = mean(stability)
        conc_var = sum((c - conc_mean) ** 2 for c in conc)
        stab_var = sum((s - stab_mean) ** 2 for s in stability)
        if conc_var == 0 or stab_var == 0:
            corr[layer] = 0.0
            continue
        cov = sum((c - conc_mean) * (s - stab_mean) for c, s in zip(conc, stability))
        corr[layer] = cov / math.sqrt(conc_var * stab_var)
    return corr


def _aggregate_layer_tokens(prompts: List[Dict]) -> Dict[int, Dict[str, int]]:
    token_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in prompts:
        for step in p.get("steps", []):
            layer = step["layer"]
            token_strs = step.get("top_token_strs") or []
            token_ids = step.get("top_token_ids") or []
            if token_strs:
                for tok in token_strs:
                    token_counts[layer][tok] += 1
            else:
                for tok_id in token_ids:
                    token_counts[layer][f"id:{tok_id}"] += 1
    return token_counts


def main():
    parser = argparse.ArgumentParser(description="Analyze entropy_report.json.")
    parser.add_argument("--report", type=str, default="entropy_report.json")
    # parser.add_argument("--report", type=str, default="qwen_entropy_report.json")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="展示 concentration 最高的前 N 个 (layer,step) 记录",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=0.75,
        help="用于判定高集中度层的分位阈值",
    )
    parser.add_argument(
        "--stable-std",
        type=float,
        default=0.05,
        help="稳定高集中度层的 std 上限",
    )
    parser.add_argument(
        "--surge-delta",
        type=float,
        default=0.1,
        help="后期突增判定阈值 (late_mean - early_mean)",
    )
    parser.add_argument(
        "--token-top-k",
        type=int,
        default=5,
        help="高集中度/高方差层的 top-k 关注 token 展示数量",
    )
    args = parser.parse_args()

    # 加载数据
    data = load_report(args.report)
    prompts = data.get("prompts", [])
    if not prompts:
        # 如果没有 prompts 字段，则文件可能错误 / 采样失败
        raise SystemExit("No prompts found in report.")

    # 聚合数据
    agg = aggregate_prompts(prompts)
    layer_step_table = _build_layer_step_table(agg)

    # 按 concentration 均值排序（降序）
    sorted_items = sorted(
        agg.items(),
        key=lambda kv: mean(kv[1].get("concentration_corrected", [])),
        reverse=True,
    )

    print(
        f"Loaded {len(prompts)} prompts. Showing top {args.top} (layer,step) by corrected concentration:\n"
    )
    for (layer, step), vals in sorted_items[: args.top]:
        print(format_row(layer, step, vals))

    # 为每一层挑选集中度最高的 step
    per_layer_best: Dict[int, Tuple[int, float]] = {}
    for (layer, step), vals in agg.items():
        conc_mean = mean(vals.get("concentration_corrected", []))
        # 若该 layer 尚未记录或当前 conc 更高，则更新
        if layer not in per_layer_best or conc_mean > per_layer_best[layer][1]:
            per_layer_best[layer] = (step, conc_mean)

    print("\nPer-layer highest concentration step:")
    for layer in sorted(per_layer_best):
        step, conc = per_layer_best[layer]
        jac_mean = mean(agg[(layer, step)].get("top_token_jaccard", []))
        print(
            f"layer={layer:02d} best_step={step:03d} conc_mean={conc:.4f} jacc_mean={jac_mean:.4f}"
        )

    # 识别稳定高集中层与后期突增层
    layer_labels = _classify_layers(
        layer_step_table,
        high_percentile=args.high_percentile,
        stable_std=args.stable_std,
        surge_delta=args.surge_delta,
    )
    print("\nLayer trend classification (by step-wise corrected concentration):")
    for layer in sorted(layer_labels):
        print(f"layer={layer:02d} label={layer_labels[layer]}")

    # concentration 与 token 稳定性的对齐关系（相关性）
    layer_corr = _layer_token_alignment(layer_step_table)
    if layer_corr:
        print("\nConcentration-token stability alignment (Pearson):")
        for layer in sorted(layer_corr):
            print(f"layer={layer:02d} corr={layer_corr[layer]:.3f}")

    # 高集中度或高方差层的 top-k token
    token_counts = _aggregate_layer_tokens(prompts)
    if token_counts:
        high_layers = [
            layer
            for layer, label in layer_labels.items()
            if label in {"stable_high", "late_surge", "high_variance"}
        ]
        if high_layers:
            print("\nTop-k attended tokens for highlighted layers:")
            for layer in sorted(high_layers):
                token_bucket = token_counts.get(layer, {})
                top_tokens = sorted(
                    token_bucket.items(), key=lambda kv: kv[1], reverse=True
                )[: args.token_top_k]
                tokens_fmt = ", ".join(f"{tok}({cnt})" for tok, cnt in top_tokens)
                print(f"layer={layer:02d} tokens={tokens_fmt}")


if __name__ == "__main__":
    main()
