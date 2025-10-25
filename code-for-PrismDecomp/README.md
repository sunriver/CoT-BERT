# PrismDecomp: 基于棱镜分解的多语义句子表示学习（SPR版本）

## 项目简介

本项目实现了基于光学类比的多语义句子表示学习方法，借鉴白光通过棱镜分解为七色光的原理，将BERT编码的句子表示通过分解器分解为7个具有不同语义的表示，并采用**自正则化（Self-Projection Regularization, SPR）**方法对每个语义表示进行并行处理。

## 核心创新

### 1. 光学类比创新 ⭐⭐⭐⭐⭐
- 将物理现象与NLP任务结合，类比新颖
- 将句子表示类比为白光，分解器类比为棱镜
- 将单一句子表示分解为7个不同语义的表示

### 2. 多语义分解创新 ⭐⭐⭐⭐
- 将单一句子表示分解为多个语义维度
- 每个维度专注于特定的语义信息
- 并行处理提高了计算效率

### 3. SPR自正则化创新 ⭐⭐⭐⭐
- **不使用InfoNCE损失**：减少负样本计算，提高训练效率
- **自正则化方法**：为每个语义表示分别计算SPR损失
- **语义空间正则化**：确保预测结果不偏离原始语义表示太多

## 语义维度定义

我们定义了7个语义维度，每个维度对应不同的语义信息：

1. **情感语义**：捕获句子的情感倾向（正面/负面/中性）
2. **主题语义**：识别句子的主题类别（科技/体育/政治等）
3. **语法语义**：理解句子的语法结构（主谓宾/定语从句等）
4. **时序语义**：处理时间关系（过去/现在/将来）
5. **空间语义**：理解空间关系（位置/方向/距离）
6. **因果语义**：识别因果关系（原因/结果/条件）
7. **程度语义**：量化程度强度（高/中/低）

## 项目结构

```
code-for-PrismDecomp/
├── prism_decomp_model.py          # 核心模型实现
├── prism_decomp_trainer.py        # 训练器实现
├── prism_decomp_train.py          # 训练脚本
├── prism_decomp_evaluation.py     # 评估脚本
├── configs/                       # 配置文件
│   ├── train_default.yaml
│   ├── train_mac_m4.yaml
│   ├── train_linux_cuda.yaml
│   └── evaluation_default.yaml
├── train.sh                       # 训练脚本
├── evaluation.sh                  # 评估脚本
├── train_mac_m4.sh               # Mac M4训练脚本
├── train_linux_cuda.sh           # Linux CUDA训练脚本
└── README.md                     # 项目说明
```

## SPR自正则化方法

### 核心思想
SPR（Self-Projection Regularization）是一种语义空间正则化方法，不需要"第二个视图"，通过自正则化确保模型内部空间的一致性。

### 实现步骤
对于每个语义表示 h_i：

1. **编码句子** → 得到表示 h_i
2. **投影** → z_i = f_proj(h_i)  
3. **预测** → p_i = f_pred(z_i)
4. **SPR正则项** → L_spr_i = ||p_i - h_i||²

### SPR损失函数
```
L_spr = ||p - h||²
```

其中：
- h: 原始语义表示
- p: 预测结果
- ||·||²: L2距离的平方

### SPR的优势
- **模型学会在内部空间保持一致**
- **不需要"第二个视图"**（避免数据增强的复杂性）
- **防止embedding全部收缩到一个点**（即塌陷collapse）

## 核心组件

### 1. SemanticDecomposer（语义分解器）
- 将句子表示分解为多个语义维度
- 使用可学习的分解矩阵
- 应用正交约束确保语义独立性
- 使用Gram-Schmidt正交化过程

### 2. SPR_Module（自正则化模块）
- **投影层 f_proj**: h → z
- **预测层 f_pred**: z → p  
- **SPR损失**: L_spr = ||p - h||²
- 不使用InfoNCE损失，减少负样本计算

### 3. MultiSemanticSPR（多语义SPR模型）
- 结合语义分解和并行SPR处理
- 7个并行的SPR模块
- 为每个语义表示分别计算SPR损失
- 融合层将处理后的表示融合为最终表示

## 使用方法

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.2+
- 其他依赖见requirements.txt

### 训练

```bash
# 默认训练
bash train.sh

# Mac M4训练
bash train_mac_m4.sh

# Linux CUDA训练
bash train_linux_cuda.sh
```

### 评估

```bash
# 评估训练好的模型
bash evaluation.sh
```

## 配置说明

### 训练配置
- `model_name_or_path`: 预训练模型路径
- `train_file`: 训练数据文件
- `output_dir`: 输出目录
- `num_semantics`: 语义维度数量（默认7）
- `orthogonal_constraint`: 是否使用正交约束
- `semantic_weights_learnable`: 语义权重是否可学习

### 评估配置
- `model_name_or_path`: 训练好的模型路径
- `do_eval`: 是否进行评估
- `eval_transfer`: 是否评估迁移任务

## 实验数据集

与code-for-SPR项目使用相同的数据集：
- **训练数据**: wiki1m_for_simcse.txt
- **评估数据**: SentEval基准测试
  - STS-B: 语义文本相似度基准
  - SICK-R: 语义相关度数据集
  - MRPC: 微软研究段落语料库
  - QQP: Quora问题对数据集
  - SNLI: 斯坦福自然语言推理数据集

## 技术特点

### 1. 理论基础
- CSE-SFP方法已经证明了单次前向传播的有效性
- 多语义分解有理论支撑
- 光学类比提供了直观的理解

### 2. 计算效率
- 相比多次前向传播，保持了效率优势
- 并行处理提高了训练速度
- 单次前向传播减少了计算开销

### 3. 语义丰富性
- 7个语义维度可能捕获更全面的语义信息
- 细粒度的语义分解提供了更多信息
- 为下游任务提供了更丰富的表示

## 预期结果

### 性能预期
- **STS-B**: 预期Spearman相关系数达到85%以上
- **SICK-R**: 预期Spearman相关系数达到80%以上
- **MRPC**: 预期F1分数达到90%以上
- **SNLI**: 预期准确率达到90%以上

### 效率指标
- **训练时间**: 相比SimCSE增加不超过20%
- **推理时间**: 相比SimCSE增加不超过10%
- **内存使用**: 相比SimCSE增加不超过30%

## 学术价值

### 创新性评估
- **理论创新**: ⭐⭐⭐⭐ 光学类比新颖，多语义分解思路有创新性
- **技术创新**: ⭐⭐⭐⭐ 分解器设计、训练策略、融合方法有技术价值
- **应用创新**: ⭐⭐⭐ 多维度语义、可解释性、效率优化有应用价值

### 发表建议
- **目标期刊**: ACL、EMNLP、NAACL、TACL
- **投稿策略**: 先投会议，充分实验，理论完善，写作优化
- **时间规划**: 实验阶段3-4个月，写作阶段1-2个月

## 参考文献

1. Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.
2. Chuang, Y. S., et al. (2022). DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings. NAACL.
3. Jiang, T., et al. (2022). PromptBERT: Improving BERT Sentence Embeddings with Prompts. ACL.
4. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.

## 联系方式

如有问题，请通过以下方式联系：
- 项目地址: [GitHub链接]
- 邮箱: [联系邮箱]

## 许可证

本项目采用MIT许可证，详见LICENSE文件。
