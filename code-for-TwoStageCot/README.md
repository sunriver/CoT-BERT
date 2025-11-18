# TwoStageCoT-BERT: 基于两阶段思维链的句子表示学习

## 项目简介

本项目实现了基于BERT的两阶段思维链分解方法，将原始长模版分解为两个短模版，通过embedding替换机制实现两阶段表示学习。**核心创新**：原始模版 "The sentence of [X] means [mask], so it can be summarized as [mask]." 太长，引入的噪声较大。因此，我们设计了**两阶段思维链分解机制**：

1. **第一阶段**：使用三种模版（负例/锚句/正例）分别获得第一阶段句子表示 h_neg, h_anchor, h_pos
   - 负例模版："The sentence of \"[X]\" doesn't mean [MASK]."
   - 锚句模版："The sentence of \"[X]\" means [MASK]."
   - 正例模版："The sentence : \"[X]\" means [MASK]."
2. **第二阶段**：将三个 h 分别注入模版2 "so [IT_SPECIAL_TOKEN] can be summarized as [MASK]."，得到三个 h_plus
   - 获取模版 embedding 矩阵
   - 将 [IT_SPECIAL_TOKEN] 位置的 token embedding 替换为第一阶段得到的 h
   - 将新的 embedding 矩阵输入到 BERT，得到第二阶段的 mask 表示 h_plus

训练阶段使用 InfoNCE 损失，正样本对为 (h_anchor_plus, h_pos_plus)，负样本为 h_neg_plus 及批次中其他句子的 h_plus。

## 核心创新

### 1. 两阶段思维链分解 ⭐⭐⭐⭐⭐
- **问题**：原始长模版 "The sentence of [X] means [mask], so it can be summarized as [mask]." 太长，引入的噪声较大
- **解决方案**：将思维链分拆成两步，并使用三种模版增强对比学习
  - **第一步**：使用三种模版分别获得第一阶段句子表示
    - 负例模版："The sentence of \"[X]\" doesn't mean [MASK]." → h_neg
    - 锚句模版："The sentence of \"[X]\" means [MASK]." → h_anchor
    - 正例模版："The sentence : \"[X]\" means [MASK]." → h_pos
  - **第二步**：将三个 h 分别注入模版2 "so [IT_SPECIAL_TOKEN] can be summarized as [MASK]."
    - 获取模版 embedding 矩阵
    - 将 [IT_SPECIAL_TOKEN] 位置的 token embedding 替换为对应的 h
    - 输入到 BERT 得到三个 h_plus (h_neg_plus, h_anchor_plus, h_pos_plus)
- **优势**：
  - 降低模版长度，减少噪声
  - 通过两阶段处理提高表示质量
  - 保持思维链的连贯性
  - 使用三种模版增强对比学习效果

### 2. Embedding 替换机制 ⭐⭐⭐⭐⭐
- **核心机制**：将第一阶段得到的句子表示 h 替换到第二阶段模版中 [IT_SPECIAL_TOKEN] 位置的 token embedding
- **实现方式**：
  1. 获取第二阶段模版的 embedding 矩阵
  2. 定位 [IT_SPECIAL_TOKEN] token 在模版中的位置
  3. 将 [IT_SPECIAL_TOKEN] 位置的 embedding 替换为 h
  4. 使用替换后的 embedding 矩阵输入到 BERT
- **优势**：
  - 实现两阶段信息的传递
  - 保持 BERT 的端到端训练
  - 无需额外的投影层

### 3. InfoNCE 对比学习 ⭐⭐⭐⭐
- **损失函数**：InfoNCE 损失
- **正样本对**：(h_anchor_plus, h_pos_plus) - 锚句与正例的第二阶段表示
- **负样本对**：h_neg_plus（负例的第二阶段表示）及批次中其他句子的 h_plus
- **优势**：
  - 简单有效的对比学习
  - 充分利用批次内负样本和负例模版
  - 提高表示质量
  - 通过三种模版增强对比学习效果

## 项目结构

```
TwoStageCoT-BERT/
├── code/
│   ├── two_stage_cot_model.py      # 核心模型实现（两阶段处理）
│   ├── two_stage_cot_trainer.py    # 训练器实现
│   ├── two_stage_cot_train.py      # 训练脚本
│   ├── configs/                     # 配置文件
│   │   ├── train_default.yaml
│   │   ├── train_mac_m4.yaml
│   │   ├── train_linux_cuda.yaml
│   │   └── evaluation_default.yaml
│   ├── train.sh                     # 训练脚本
│   ├── evaluation.sh                # 评估脚本
│   ├── parse_args_util.py           # 参数解析工具
│   ├── lmf_log_util.py               # 日志工具
│   ├── platform_utils.py             # 跨平台工具
│   └── README.md                     # 项目说明
├── data/                             # 数据目录
└── requirements.txt                  # 依赖文件
```

## 核心组件

### 1. 两阶段模版处理

#### 第一阶段模版（三种）
- **负例模版**：`"The sentence of \"[X]\" doesn't mean [MASK]."`
  - 作用：获得负例句子表示 h_neg
- **锚句模版**：`"The sentence of \"[X]\" means [MASK]."`
  - 作用：获得锚句句子表示 h_anchor
- **正例模版**：`"The sentence : \"[X]\" means [MASK]."`
  - 作用：获得正例句子表示 h_pos
- **表示提取**：从 [MASK] 位置的 BERT 隐藏状态提取句子语义表示

#### 第二阶段模版
- **模版格式**：`"so [IT_SPECIAL_TOKEN] can be summarized as [MASK]."`
- **作用**：获得第二阶段句子表示 h+
- **处理流程**：
  1. 获取模版 embedding 矩阵
  2. 定位 [IT_SPECIAL_TOKEN] token 位置
  3. 将 [IT_SPECIAL_TOKEN] 位置的 embedding 替换为 h
  4. 输入到 BERT 得到 h+

### 2. Embedding 替换机制

**实现步骤**：
1. 获取第二阶段模版的 token ids（不包含特殊token）
2. 找到 [IT_SPECIAL_TOKEN] token 在模版中的位置
3. 构建完整的第二阶段输入（添加特殊token）
4. 通过 embedding 层获取模版的 embedding
5. 将 h 替换到 [IT_SPECIAL_TOKEN] 位置的 embedding
6. 使用替换后的 embedding 输入到 BERT

### 3. InfoNCE 损失计算

**损失函数**：
```
L = InfoNCE(h_anchor_plus, h_pos_plus, h_neg_plus, h_plus_batch)
```

其中：
- **正样本对**：(h_anchor_plus[i], h_pos_plus[i]) - 同一句子的锚句与正例第二阶段表示
- **负样本对**：
  - h_neg_plus[i] - 同一句子的负例第二阶段表示
  - h_plus_batch[j] - 批次中其他句子的所有 h_plus (j ≠ i)

**计算方式**：
1. 归一化所有 h_plus 表示
2. 计算正样本对相似度 (h_anchor_plus, h_pos_plus)
3. 构建负样本候选池：包含 h_neg_plus 和批次中所有其他句子的 h_plus
4. 计算锚句与所有候选的相似度
5. 排除自身（锚句和正例）后，组合相似度矩阵
6. 使用 CrossEntropyLoss 计算对比损失

## 使用方法

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.2+
- 其他依赖见 requirements.txt

### 训练

```bash
# 默认训练
bash train.sh

# 使用自定义配置文件
python two_stage_cot_train.py configs/train_default.yaml
```

### 评估

```bash
# 评估训练好的模型
bash evaluation.sh

# 使用自定义配置文件
python two_stage_cot_train.py configs/evaluation_default.yaml
```

## 配置说明

### 训练配置

- `model_name_or_path`: 预训练模型路径
- `train_file`: 训练数据文件
- `output_dir`: 输出目录
- `temperature`: InfoNCE损失温度参数（默认0.05）
- `stage1_negative_template`: 第一阶段负例模版（默认 "The sentence of \"[X]\" doesn't mean [MASK]."）
- `stage1_anchor_template`: 第一阶段锚句模版（默认 "The sentence of \"[X]\" means [MASK]."）
- `stage1_positive_template`: 第一阶段正例模版（默认 "The sentence : \"[X]\" means [MASK]."）
- `stage2_template`: 第二阶段模版（默认 "so [IT_SPECIAL_TOKEN] can be summarized as [MASK]."）

### 评估配置

- `model_name_or_path`: 训练好的模型路径
- `do_eval`: 是否进行评估
- `eval_transfer`: 是否评估迁移任务

## 实验数据集

- **训练数据**: wiki1m_for_simcse.txt
- **评估数据**: SentEval基准测试
  - STS-B: 语义文本相似度基准
  - SICK-R: 语义相关度数据集
  - MRPC: 微软研究段落语料库
  - QQP: Quora问题对数据集
  - SNLI: 斯坦福自然语言推理数据集

## 技术特点

### 1. 理论基础
- 两阶段思维链分解有理论支撑
- Embedding 替换机制保持端到端训练
- InfoNCE 对比学习提高表示质量

### 2. 计算效率
- 相比原始长模版，减少噪声
- 两阶段处理提高表示质量
- 保持 BERT 的端到端训练

### 3. 表示质量
- 两阶段处理捕获更丰富的语义信息
- Embedding 替换实现信息传递
- 为下游任务提供更好的表示

## 预期结果

### 性能预期
- **STS-B**: 预期 Spearman 相关系数达到 85% 以上
- **SICK-R**: 预期 Spearman 相关系数达到 80% 以上
- **MRPC**: 预期 F1 分数达到 90% 以上
- **SNLI**: 预期准确率达到 90% 以上

### 效率指标
- **训练时间**: 相比原始长模版方法，训练时间略有增加（两阶段处理）
- **推理时间**: 相比原始长模版方法，推理时间略有增加
- **内存使用**: 相比原始长模版方法，内存使用基本一致

## 学术价值

### 创新性评估
- **理论创新**: ⭐⭐⭐⭐ 两阶段思维链分解思路有创新性
- **技术创新**: ⭐⭐⭐⭐ Embedding 替换机制、两阶段处理有技术价值
- **应用创新**: ⭐⭐⭐ 降低噪声、提高表示质量有应用价值

### 发表建议
- **目标期刊**: ACL、EMNLP、NAACL、TACL
- **投稿策略**: 先投会议，充分实验，理论完善，写作优化
- **时间规划**: 实验阶段3-4个月，写作阶段1-2个月

## 参考文献

1. Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.
2. Jiang, T., et al. (2022). PromptBERT: Improving BERT Sentence Embeddings with Prompts. ACL.
3. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
4. Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.

## 联系方式

如有问题，请通过以下方式联系：
- 项目地址: [GitHub链接]
- 邮箱: [联系邮箱]

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

