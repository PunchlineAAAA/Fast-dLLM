import json

path = "/home/jovyan/workspace/Fast-dLLM/v2/results/minerva_math/Efficient-Large-Model__Fast_dLLM_v2_7B/samples_minerva_math_precalc_2025-12-04T09-56-36.223451.jsonl"

wrong_list = []
total = 0
correct = 0

with open(path) as f:
    for line in f:
        item = json.loads(line)

        # 统计所有样本（Minerva 评测不使用 flexible-extract）
        total += 1

        # Math correctness metric
        math_ok = item.get("math_verify", 0) == 1

        if math_ok:
            correct += 1
            continue

        # ---------- 新版 doc 结构 ----------
        doc = item["doc"]
        problem = doc.get("problem", "")
        gold_answer = doc.get("answer", "")
        solution = doc.get("solution", "")
        # ------------------------------------

        # LM-Eval 的 standard target
        target = item.get("target", "")

        # 模型最终匹配后答案（filtered_resps）
        pred_clean = (
            item["filtered_resps"][0]
            if item.get("filtered_resps") and len(item["filtered_resps"]) > 0
            else ""
        )

        # 模型原始输出
        model_output = (
            item["resps"][0][0]
            if item.get("resps") and len(item["resps"]) > 0
            else ""
        )

        wrong_list.append({
            "doc_id": item["doc_id"],
            "problem": problem,
            "gold_answer": gold_answer,
            "lm_eval_target": target,
            "solution": solution,

            "model_pred_clean": pred_clean,
            "model_output": model_output,

            "exact_match": item.get("exact_match", None),
            "math_verify": item.get("math_verify", None),
        })

# 保存错误
with open("wrong_questions.jsonl", "w") as f:
    for w in wrong_list:
        f.write(json.dumps(w, ensure_ascii=False) + "\n")

with open("wrong_questions.txt", "w") as f:
    for w in wrong_list:
        f.write(f"doc_id: {w['doc_id']}\n")
        f.write(f"题目: {w['problem']}\n")
        f.write(f"正确答案(answer): {w['gold_answer']}\n")
        f.write(f"LM-eval target:\n{w['lm_eval_target']}\n")
        f.write(f"模型预测（清洗后）: {w['model_pred_clean']}\n")
        f.write(f"原始输出:\n{w['model_output']}\n")
        f.write(f"math_verify: {w['math_verify']} | exact_match: {w['exact_match']}\n")
        f.write(f"解析(solution):\n{w['solution']}\n")
        f.write("-" * 60 + "\n")

wrong = len(wrong_list)
acc = correct / total * 100 if total > 0 else 0.0

print(f"总题数: {total}")
print(f"正确: {correct}")
print(f"错误: {wrong}")
print(f"正确率（基于 math_verify）: {acc:.2f}%")
print("已保存到 wrong_questions.jsonl 和 wrong_questions.txt")
