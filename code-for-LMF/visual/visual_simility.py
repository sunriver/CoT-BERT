
import os

current_working_directory = os.getcwd()
print(f"当前工作目录: {current_working_directory}")

# ==============================================================================
# 步骤 0: 设置 Colab 环境
# 确保运行时类型为 GPU (代码执行程序 -> 更改运行时类型)
# ==============================================================================

# ==============================================================================
# 步骤 1: 安装必要的库
# ==============================================================================

# 导入所需的库
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
# from sentence_transformers import SentenceTransformer, util
import torch # 用于设备管理
# from datasets import disable_caching
# disable_caching()  # 强制刷新缓存
# ==============================================================================
# 步骤 2: 加载 STS-B 测试集
# ==============================================================================
# from datasets import load_dataset

print("\nStep 2: Loading STS-B test set...")
# 加载 STS-B 数据集
# 'sts_benchmark' 是 Hugging Face datasets 中的数据集名称
# split='test' 明确指定使用测试集
# sts_dataset = load_dataset("stsb", "test")
# 加载单个CSV文件

# from datasets import load_dataset

# 加载CSV文件（处理STS测试集）
# import pandas as pd
# sts_dataset = pd.read_csv("/workspace/CoT-BERT/SentEval/data/downstream/STS/STSBenchmark/sts-test.csv", sep=",", error_bad_lines=False,      # 跳过格式错误行（谨慎使用）
#         warn_bad_lines=True , encoding="utf-8")
# print(df.head())

# from lmf_model import BertForCL
# model_name_or_path = "/workspace/CoT-BERT/result/CoT-LMF"
# config = AutoConfig.from_pretrained(model_name_or_path)
# model = BertForCL.from_pretrained(model_name_or_path, cofig = config)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)


import sys
# Import SentEval
# Set PATHs
PATH_TO_SENTEVAL = '/workspace/CoT-BERT/SentEval'
# PATH_TO_DATA = '/workspace/CoT-BERT/SentEval/data'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
data_path = "/workspace/CoT-BERT/SentEval/data/downstream/STS/STSBenchmark/"
sTSBenchmarkFinetune = senteval.sts.STSBenchmarkFinetune(task_path = data_path)
sts_dataset = sTSBenchmarkFinetune.sick_data['test']

# sts_dataset = load_dataset(
#     path='csv',
#     data_files='/workspace/CoT-BERT/SentEval/data/downstream/STS/STSBenchmark/sts-test.csv',
# )

# sts_dataset = load_dataset("nyu-mll/glue", "stsb", split="test")

# 提取句子对和黄金标准相似度分数
sentences1 = sts_dataset['X_A']
sentences2 = sts_dataset['X_B']
scores = sts_dataset['y']
gold_scores = np.array(scores)

print(f"Loaded {len(sentences1)} sentence pairs from STS-B test set.")
print(f"Example: S1='{sentences1[0]}', S2='{sentences2[0]}', Gold Score={gold_scores[0]}")

# ==============================================================================
# 步骤 3: 定义模型并生成嵌入及相似度预测
# ==============================================================================
print("\nStep 3: Defining models and generating embeddings & similarity predictions...")

cos_sim = torch.nn.CosineSimilarity(dim=-1)

# 定义一个函数来生成嵌入并计算余弦相似度
def get_similarity_predictions(model, sentences1, sentences2, gold_scores, use_cls_token=False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if  model is not None:
         model.to(device)

    sentences1 = [' '.join(s) for s in sentences1]
    sentences2 = [' '.join(s) for s in sentences2]

    # 对于 SentenceTransformer，默认是平均池化
    # 如果要模拟 [CLS] token，我们需要手动处理
    if use_cls_token:
        print("Using [CLS] token embeddings (custom handling)...")
        # 需要加载原始的 transformers 模型才能直接获取 [CLS] token
        # 这比 SentenceTransformer 的高级API复杂，但为了模拟原始 BERT-base [CLS]

        pretrain_path = "/workspace/pretrain_models/bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        base_bert_model = AutoModel.from_pretrained(pretrain_path).to(device)

        # all_sentences = sentences1 + sentences2
        # 分批处理以避免内存问题
        batch_size = 32
        embeddings = []
        for i in range(0, len(sentences1), batch_size):
            batch_sentences1 = sentences1[i:i+batch_size]
            batch_sentences2 = sentences2[i:i+batch_size]
            encoded_input1 = tokenizer(batch_sentences1, padding=True, truncation=True, return_tensors='pt').to(device)
            encoded_input2 = tokenizer(batch_sentences2, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output1 = base_bert_model(**encoded_input1)
                model_output2 = base_bert_model(**encoded_input2)
            # 提取 [CLS] token 的 embedding
            embeddings.append(model_output1.last_hidden_state[:, 0, :].cpu().numpy()) # [CLS] token is the first token
            embeddings.append(model_output2.last_hidden_state[:, 0, :].cpu().numpy()) # [CLS] token is the first token

        all_embeddings = np.vstack(embeddings)
        embeddings1 = all_embeddings[:len(sentences1)]
        embeddings2 = all_embeddings[len(sentences1):]

    else:
        # 使用 SentenceTransformer 的 encode 方法 (默认是平均池化或其他训练好的池化策略)
        print("Using SentenceTransformer default pooling (e.g., mean pooling)...")
        embeddings1 = model.encode(sentences1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True, device=device)
        embeddings2 = model.encode(sentences2, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True, device=device)

    # 计算余弦相似度
    # 转换为张量
    tensor1 = torch.from_numpy(embeddings1).float().to(device)
    tensor2 = torch.from_numpy(embeddings2).float().to(device)
    cosine_similarities = cos_sim(tensor1, tensor2)
    # 将 torch.Tensor 转换为 numpy 数组，并展平
    predicted_similarities = cosine_similarities.diag().cpu().numpy()
    predicted_similarities = [predicted_similarities[i][i] for i in range(len(predicted_similarities))]


    # 打印相似性
    for i in range(len(predicted_similarities)):
        sent1 = sentences1[i]
        sent2 = sentences2[i]
        simility = predicted_similarities[i]
        gold_score = gold_scores[i]
        print(f"{sent1} -----> {sent2}: {simility}, {gold_score}")

    # 归一化预测相似度到 0-1 范围，因为余弦相似度本身就在这个范围
    # STS-B 的黄金标准是 0-5，但我们的预测相似度是 0-1
    # 我们可以选择不进行缩放，或者进行线性缩放到 0-5 来匹配黄金标准，
    # 但由于图中的预测余弦相似度也是 0-1，我们保持 0-1。
    # 如果黄金标准是 0-5，预测值是 0-1，图表仍然能正确显示相关性。
    return predicted_similarities


# --- 定义并评估三个模型 ---

# Model 1: RoBERTa-base ([CLS]) - 模拟原始 BERT [CLS] 效果差的情况
# 使用 'bert-base-uncased' 作为基底，手动提取 [CLS]
# 注意: SentenceTransformer 库默认会对 [CLS] 进行池化，
# 要模拟原始 [CLS] 行为，需要更底层的 Transformers 库操作
print("\n--- Evaluating bert-base-uncased ([CLS]) ---")
# 这里我们用一个变通方法：加载 bert-base-uncased，然后手动获取 CLS token
# 这会和 SentenceTransformer 的模型有些不同，但能更好地模拟 CLS 的低性能
# 实际上，使用 sentence-transformers/bert-base-nli-mean-tokens 的 mean pooling
# 比 bert-base-uncased CLS token 性能要好很多
# 为了模拟图中的“差”效果，我们特意使用未经 NLI 微调的 bert-base-uncased CLS token
model1_name = "bert-base-uncased ([CLS])" # 对应论文中的概念，但我们用bert-base-uncased的CLS实现
predicted_similarities1 = get_similarity_predictions(
    model=None, # 这里模型是 None，因为我们在函数内部手动加载 transformers 的 base 模型
    sentences1=sentences1,
    sentences2=sentences2,
    gold_scores=gold_scores,
    use_cls_token=True
)





# Model 2: RoBERTa-base + PromptBERT ([MASK]) - 模拟中等性能的模型
# 使用一个在 STS-B 上微调过的 RoBERTa-base 模型来模拟 PromptBERT 的效果
# print("\n--- Evaluating RoBERTa-base + PromptBERT (Simulated) ---")
# model2_name = "RoBERTa-base + PromptBERT (Simulated)"
# model2 = SentenceTransformer('stsb-roberta-base') # 在 STS-B 上微调过的 RoBERTa-base
# predicted_similarities2 = get_similarity_predictions(model2, sentences1, sentences2, gold_scores)




# Model 3: RoBERTa-base + CoT-BERT ([MASK]) - 模拟最佳性能的模型
# 使用一个在 STS-B 上微调过的 RoBERTa-large 模型来模拟 CoT-BERT 的效果
# print("\n--- Evaluating RoBERTa-base + CoT-BERT (Simulated) ---")
# model3_name = "RoBERTa-base + CoT-BERT (Simulated)"
# model3 = SentenceTransformer('stsb-roberta-large') # 在 STS-B 上微调过的 RoBERTa-large
# predicted_similarities3 = get_similarity_predictions(model3, sentences1, sentences2, gold_scores)


# ==============================================================================
# 步骤 4: 绘制散点图
# ==============================================================================
print("\nStep 4: Plotting results...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

# 绘制 Model 1
axes[0].scatter(gold_scores, predicted_similarities1, alpha=0.2, s=5, color='blue')
axes[0].set_title(model1_name, fontsize=14)
axes[0].set_xlabel('the gold similarity score', fontsize=12)
axes[0].set_ylabel('the predicted cosine similarity', fontsize=12)
axes[0].set_xlim(-0.5, 5.5) # STS-B 分数范围
axes[0].set_ylim(-0.05, 1.05) # 余弦相似度范围
axes[0].grid(True, linestyle='--', alpha=0.6)

# 绘制 Model 2
# axes[1].scatter(gold_scores, predicted_similarities2, alpha=0.2, s=5, color='red')
# axes[1].set_title(model2_name, fontsize=14)
# axes[1].set_xlabel('the gold similarity score', fontsize=12)
# # axes[1].set_ylabel('the predicted cosine similarity', fontsize=12) # 共享Y轴，不重复设置
# axes[1].grid(True, linestyle='--', alpha=0.6)

# 绘制 Model 3
# axes[2].scatter(gold_scores, predicted_similarities3, alpha=0.2, s=5, color='green')
# axes[2].set_title(model3_name, fontsize=14)
# axes[2].set_xlabel('the gold similarity score', fontsize=12)
# # axes[2].set_ylabel('the predicted cosine similarity', fontsize=12) # 共享Y轴，不重复设置
# axes[2].grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Predicted vs Gold Similarity on STS-B Test Set', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以避免标题重叠

# 在绘图代码后添加
plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
print("图像已保存为 'comparison_plot.png'")

plt.show()

print("\nExperiment complete. Check the plot above.")