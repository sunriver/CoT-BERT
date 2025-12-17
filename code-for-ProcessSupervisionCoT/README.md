# ProcessSupervisionCoT-BERT: 基于过程监督的 CoT-BERT 句子表示学习

## 项目简介

本项目实现了基于过程监督的 CoT-BERT 方法，在 CoT-BERT 基础上，保留三模板 InfoNCE 损失，加入跨模板对比学习约束相同位置 mask 相似，并对 mask1 和 mask2 分别进行过程监督。**核心创新**：

1. **三模板 InfoNCE 损失**：保留 CoT-BERT 原有的对比学习功能，使用融合后的 mask1 和 mask2 表示进行对比
2. **跨模板对比学习**：约束锚句模板和正样本模板的相同位置 mask 表示相似（mask1 相似，mask2 也相似）
3. **过程监督**：对 mask1 和 mask2 的推理路径分别进行质量评估，使用 LLM（GPT-4o-mini）评估推理路径的逻辑性、完整性和简洁性

## 核心创新

### 1. 三模板 InfoNCE 对比学习 ⭐⭐⭐⭐

- **模板设计**：使用三种 CoT 模板（锚句、正样本、负样本）
  - 锚句模板：`The sentence: "[X]" means [MASK1], so it can be summarized as [MASK2].`
  - 正样本模板：`The sentence: "[X]" means [MASK1], so it can be summarized as [MASK2].`
  - 负样本模板：`The sentence: "[X]" does not mean [MASK1], so it cannot be summarized as [MASK2].`
- **表示提取**：分别提取 mask1 和 mask2 的表示，然后融合为句子表示
- **损失函数**：使用 InfoNCE 损失，正样本对为锚句和正样本的融合表示，负样本为负样本模板及批次内其他样本

### 2. 跨模板对比学习 ⭐⭐⭐⭐⭐

- **核心机制**：约束锚句模板和正样本模板的相同位置 mask 表示相似
  - mask1_anchor 与 mask1_positive 相似
  - mask2_anchor 与 mask2_positive 相似
- **损失函数**：使用交叉 InfoNCE（icnce）损失，参考 TNCSE 的实现
- **优势**：
  - 增强模板间的一致性
  - 提高表示质量
  - 通过跨模板约束学习更鲁棒的表示

### 3. 过程监督 ⭐⭐⭐⭐⭐

- **评估机制**：使用 LLM（GPT-4o-mini）评估 mask1 和 mask2 的推理路径质量
  - **逻辑性**：推理过程是否符合逻辑
  - **完整性**：是否捕获了句子的关键信息
  - **简洁性**：是否避免了冗余信息
- **损失函数**：将评估分数（1-100）转换为过程监督损失，分数越高损失越小
- **优势**：
  - 直接监督推理过程的质量
  - 提高模型的可解释性
  - 通过过程监督学习更好的表示

## 项目结构

```
code-for-ProcessSupervisionCoT/
├── process_supervision_cot_model.py      # 核心模型实现
├── process_supervision_cot_trainer.py    # 训练器实现
├── process_supervision_cot_train.py      # 训练脚本
├── process_supervision_cot_evaluation.py  # 评估脚本
├── process_supervision_cot_supervisor.py # 过程监督评估器
├── configs/                               # 配置文件目录
│   ├── train_default.yaml
│   ├── train_linux_cuda.yaml
│   ├── train_mac_m4.yaml
│   ├── evaluation_default.yaml
│   └── evaluation_linux_cuda.yaml
├── train.sh                               # 训练启动脚本
├── evaluation.sh                          # 评估启动脚本
├── parse_args_util.py                     # 参数解析工具
├── lmf_log_util.py                        # 日志工具
├── platform_utils.py                      # 跨平台工具
├── README.md                              # 项目说明文档
└── requirements.txt                       # 依赖文件
```

## 核心组件

### 1. 模型实现 (`process_supervision_cot_model.py`)

- **BertForProcessSupervisionCoT**：主模型类
  - 支持 mask1 和 mask2 分别提取
  - 实现三种损失函数：原有 InfoNCE、跨模板对比学习、过程监督
- **损失函数**：
  - `compute_original_infonce_loss`：原有的三模板 InfoNCE 损失
  - `cross_template_icnce_loss`：跨模板对比学习损失
  - `compute_process_supervision_loss`：过程监督损失

### 2. 过程监督模块 (`process_supervision_cot_supervisor.py`)

- **ProcessSupervisor**：过程监督评估器类
  - 使用 LLM API 评估推理路径质量
  - 支持 mask1 和 mask2 分别评估
- **推理路径提取**：从输入中提取 mask1 和 mask2 位置的上下文

### 3. 训练器 (`process_supervision_cot_trainer.py`)

- **ProcessSupervisionCoTTrainer**：继承 Trainer 类
  - 实现 `compute_loss` 方法，调用模型的前向传播
  - 实现 `evaluate` 方法，支持 SentEval 评估

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
python process_supervision_cot_train.py configs/train_default.yaml
```

### 评估

```bash
# 评估训练好的模型
bash evaluation.sh

# 使用自定义配置文件
python process_supervision_cot_evaluation.py configs/evaluation_default.yaml
```

## 配置说明

### 训练配置

- `model_name_or_path`: 预训练模型路径
- `train_file`: 训练数据文件
- `output_dir`: 输出目录
- `temperature`: InfoNCE损失温度参数（默认0.05）
- `mask_embedding_sentence_template`: 锚句模板
- `mask_embedding_sentence_different_template`: 正样本模板
- `mask_embedding_sentence_negative_template`: 负样本模板
- `mask_num`: mask 数量（默认2，即 mask1 和 mask2）
- `cross_template_weight`: 跨模板对比学习损失权重（默认0.5）
- `process_supervision_weight`: 过程监督损失权重（默认0.1）
- `enable_process_supervision`: 是否启用过程监督（默认False）

### 评估配置

- `model_name_or_path`: 训练好的模型路径
- `do_eval`: 是否进行评估
- `eval_transfer`: 是否评估迁移任务
- `task_set`: 任务集合（sts, transfer, full）
- `mode`: 评估模式（dev, test, fasttest）

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

- 三模板 InfoNCE 对比学习有理论支撑
- 跨模板对比学习增强表示一致性
- 过程监督提高推理路径质量

### 2. 计算效率

- 相比原始 CoT-BERT，增加了跨模板约束和过程监督
- 过程监督默认关闭，需要时通过配置开启（避免训练时频繁调用 LLM API）
- 保持 BERT 的端到端训练

### 3. 表示质量

- 三模板对比学习捕获更丰富的语义信息
- 跨模板约束提高表示一致性
- 过程监督提高推理路径质量

## 预期结果

### 性能预期

- **STS-B**: 预期 Spearman 相关系数达到 85% 以上
- **SICK-R**: 预期 Spearman 相关系数达到 80% 以上
- **MRPC**: 预期 F1 分数达到 90% 以上
- **SNLI**: 预期准确率达到 90% 以上

### 效率指标

- **训练时间**: 相比原始 CoT-BERT，训练时间略有增加（跨模板约束和过程监督）
- **推理时间**: 相比原始 CoT-BERT，推理时间基本一致
- **内存使用**: 相比原始 CoT-BERT，内存使用基本一致

## 学术价值

### 创新性评估

- **理论创新**: ⭐⭐⭐⭐ 跨模板对比学习和过程监督思路有创新性
- **技术创新**: ⭐⭐⭐⭐ 跨模板约束、过程监督有技术价值
- **应用创新**: ⭐⭐⭐ 提高表示质量和推理路径质量有应用价值

### 发表建议

- **目标期刊**: ACL、EMNLP、NAACL、TACL
- **投稿策略**: 先投会议，充分实验，理论完善，写作优化
- **时间规划**: 实验阶段3-4个月，写作阶段1-2个月

## 参考文献

1. Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.
2. Jiang, T., et al. (2022). PromptBERT: Improving BERT Sentence Embeddings with Prompts. ACL.
3. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
4. Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.

## 注意事项

1. **mask_num 配置**：确保所有配置中 `mask_num=2`
2. **模板格式**：使用两个 mask token（*mask*）的模板
3. **损失权重**：需要调优 `cross_template_weight` 和 `process_supervision_weight`
4. **过程监督**：默认关闭，需要时通过配置开启（避免训练时频繁调用 LLM API）
5. **跨平台支持**：使用 `platform_utils.py` 确保跨平台兼容性
6. **代码复用**：尽量复用 TwoStageCoT 的工具文件，减少重复代码

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

