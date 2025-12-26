"""
Utility to analyze entropy_report.json produced by attention_entropy_probe.py.

它会对 attention_entropy_probe.py 产生的 entropy_report.json 进行分析，
并按 layer / step 总结以下指标：
- concentration（注意力集中度）
- top_token_jaccard（Top Token 集合的 Jaccard 相似度）
- head_variance（多头注意力方差）

Usage:
    python v2/entropy_report_analyzer.py --report entropy_report.json --top 3
"""

import argparse
import json
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
            "top_token_jaccard": s.get("top_token_jaccard"),
            "head_variance": s.get("head_variance"),
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
                if v is not None:
                    agg[key][metric].append(v)
    return agg


def mean(xs: List[float]) -> float:
    """计算平均值（避免空列表）"""
    return sum(xs) / len(xs) if xs else 0.0


def format_row(layer: int, step: int, vals: Dict[str, List[float]]) -> str:
    """格式化输出行，展示各类指标的均值和样本数量"""
    conc = vals.get("concentration", [])
    jac = vals.get("top_token_jaccard", [])
    var = vals.get("head_variance", [])
    return (
        f"layer={layer:02d} step={step:03d} "
        f"conc_mean={mean(conc):.4f} conc_n={len(conc)} "
        f"jacc_mean={mean(jac):.4f} jacc_n={len(jac)} "
        f"head_var_mean={mean(var):.4f}"
    )


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
    args = parser.parse_args()

    # 加载数据
    data = load_report(args.report)
    prompts = data.get("prompts", [])
    if not prompts:
        # 如果没有 prompts 字段，则文件可能错误 / 采样失败
        raise SystemExit("No prompts found in report.")

    # 聚合数据
    agg = aggregate_prompts(prompts)

    # 按 concentration 均值排序（降序）
    sorted_items = sorted(
        agg.items(),
        key=lambda kv: mean(kv[1].get("concentration", [])),
        reverse=True,
    )

    print(
        f"Loaded {len(prompts)} prompts. Showing top {args.top} (layer,step) by concentration:\n"
    )
    for (layer, step), vals in sorted_items[: args.top]:
        print(format_row(layer, step, vals))

    # 为每一层挑选集中度最高的 step
    per_layer_best: Dict[int, Tuple[int, float]] = {}
    for (layer, step), vals in agg.items():
        conc_mean = mean(vals.get("concentration", []))
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


if __name__ == "__main__":
    main()