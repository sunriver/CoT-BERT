from biosenteval import load_model

# 加载中文医学预训练模型（如 BioBERT）
model = load_model("biobert-base-chinese-cblue")

from biosenteval import evaluate

results = evaluate(
    model=model,         # 预训练模型
    task="sts",           # 任务类型（如 sts/sentence_similarity）
    data_path="./cblue_data/sts",  # 数据集路径
    output_dir="./results"  # 结果保存目录
)

print(results)  # 输出示例：{"accuracy": 0.85, "f1_score": 0.82}