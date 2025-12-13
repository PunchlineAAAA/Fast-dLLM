import json

path = "/home/jovyan/workspace/Fast-dLLM/v2/results/gsm8k/Efficient-Large-Model__Fast_dLLM_v2_7B/samples_gsm8k_2025-12-04T13-24-55.583982.jsonl"

wrong_list = []
total = 0
correct = 0

with open(path) as f:
    for line in f:
        item = json.loads(line)

        # 只统计 flexible-extract 的结果，和你之前分析保持一致
        if item.get("filter") != "flexible-extract":
            continue

        total += 1

        # eval 已经帮你算好了 per-sample 的 exact_match
        em = item.get("exact_match", 0.0)  # 通常是 0.0 或 1.0

        if em:
            correct += 1
        else:
            # 记录错误样本
            doc = item["doc"]
            question = doc["question"]
            gold = item["target"]          # 标准答案（原始文本）
            pred_clean = (
                item["filtered_resps"][0]
                if item.get("filtered_resps")
                and len(item["filtered_resps"]) > 0
                and len(item["filtered_resps"][0]) > 0
                else ""
            )
            model_output = (
                item["resps"][0][0]
                if item.get("resps")
                and len(item["resps"]) > 0
                and len(item["resps"][0]) > 0
                else ""
            )

            wrong_list.append({
                "doc_id": item["doc_id"],
                "question": question,
                "gold": gold,
                "pred": pred_clean,       # lm-eval 用来判分的那个“提取后的答案”
                "model_output": model_output,  # 原始完整输出
            })

# 保存 JSONL
with open("wrong_questions.jsonl", "w") as f:
    for w in wrong_list:
        f.write(json.dumps(w, ensure_ascii=False) + "\n")

# 保存 TXT
with open("wrong_questions.txt", "w") as f:
    for w in wrong_list:
        f.write(f"doc_id: {w['doc_id']}\n")
        f.write(f"题目: {w['question']}\n")
        f.write(f"预测(提取后): {w['pred']}\n")
        f.write(f"标准答案: {w['gold']}\n")
        f.write(f"原始输出: {w['model_output']}\n")
        f.write("-" * 60 + "\n")

wrong = len(wrong_list)
acc = correct / total * 100 if total > 0 else 0.0

print(f"总题数（flexible-extract）: {total}")
print(f"正确: {correct}")
print(f"错误: {wrong}")
print(f"正确率: {acc:.2f}%")
print("已保存到 wrong_questions.jsonl 和 wrong_questions.txt")
