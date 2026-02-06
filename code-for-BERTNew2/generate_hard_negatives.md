# 计划：BERTNew2 硬负样本生成

## 1. 目标

在 `code-for-BERTNew2` 项目结构中直接实现硬负样本生成逻辑，以便为模型训练准备数据。

## 2. 设计方案

* **位置：** `CoT-BERT/code-for-BERTNew2/generate_hard_negatives.py`
* **模板：** `[CLS] The sentence "{sent_0}" does not mean the sentence "{sent_0_mask}" [SEP]`
* **掩码策略：**
  * 20% 随机掩码。
  * 至少掩码一个 Token。
* **模型：** `BertForMaskedLM` (bert-base-uncased)。

## 3. 实现细节

* **依赖库：** `torch`, `transformers`, `tqdm`。
* **输入：** 原始语料库文件路径。
* **输出：**
  * 格式：`original_sentence \t hard_negative_sentence`
  * 每行两列，用制表符 `\t` 分隔。
* **执行逻辑：**
  1. 加载模型与分词器（Tokenizer）。
  2. 读取输入文件。
  3. 批处理循环：
     * 对 `sent_0` 进行分词。
     * 创建 `sent_0_mask`。
     * 构建模板输入序列。
     * 执行模型推理。
     * 解码并提取 `sent_0_mask` 部分的预测结果。
  4. 将结果写入输出文件。

## 4. 执行说明

用户可以在开始主训练循环之前运行此脚本来生成训练数据。
