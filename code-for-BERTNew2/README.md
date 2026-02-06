# CoT-BERT: 基于Chain-of-Thought的句子表示学习

## 项目简介

本项目实现了基于BERT的Chain-of-Thought（思维链）句子表示学习方法。通过设计特殊的模板，将句子表示学习任务转化为一个思维链推理过程，从而获得更好的句子语义表示。

## 核心创新

### 1. Chain-of-Thought模板设计 ⭐⭐⭐⭐⭐
- **核心思想**：将句子表示学习转化为思维链推理过程
- **模板格式**：`"The sentence of \"[X]\" means [MASK], so it can be summarized as [MASK]."`
- **优势**：
  - 通过思维链引导模型学习更丰富的语义信息
  - 两个MASK位置分别捕获不同层次的语义表示
  - 通过"means"和"summarized"两个步骤实现语义推理

### 2. 多模板对比学习 ⭐⭐⭐⭐
- **锚句模板**：`"The sentence of \"[X]\" means [MASK], so it can be summarized as [MASK]."`
- **正例模板**：`"The sentence : \"[X]\" means [MASK], so it can be summarized as [MASK]."`
- **负例模板**：`"The sentence : \"[X]\" does not mean [MASK], so it cannot be summarized as [MASK]."`
- **优势**：
  - 通过不同模板增强对比学习效果
  - 负例模板提供硬负样本
  - 提高模型的判别能力

### 3. Delta去噪机制 ⭐⭐⭐⭐
- **核心机制**：去除MASK位置表示中的位置相关噪声
- **实现方式**：
  1. 使用空模板计算位置相关的噪声
  2. 从MASK表示中减去该噪声
  3. 获得更纯净的语义表示
- **优势**：
  - 减少位置偏差对表示的影响
  - 提高表示的语义质量
  - 改善下游任务性能

## 项目结构

```
code-for-BERTNew/
├── cot_bert_train.py              # 训练脚本
├── cot_bert_evaluation.py         # 评估脚本
├── platform_utils.py              # 跨平台工具
├── parse_args_util.py             # 参数解析工具
├── lmf_log_util.py                # 日志工具
├── configs/                        # 配置文件目录
│   ├── train_default.yaml
│   ├── train_mac_m4.yaml
│   ├── train_linux_cuda.yaml
│   ├── evaluation_default.yaml
│   ├── evaluation_mac_m4.yaml
│   └── evaluation_linux_cuda.yaml
├── train.sh                        # 通用训练脚本
├── train_linux_cuda.sh             # Linux CUDA训练脚本
├── train_mac_m4.sh                 # Mac M4训练脚本
├── evaluation.sh                   # 通用评估脚本
├── evaluate_linux_cuda.sh          # Linux CUDA评估脚本
├── evaluate_mac_m4.sh              # Mac M4评估脚本
├── requirement_cuda_python390.txt  # 依赖文件
└── README.md                        # 项目说明
```

## 核心组件

### 1. 模型架构
- **基础模型**：BERT-base-uncased 或 RoBERTa-base
- **模板处理**：支持多个MASK位置的模板
- **表示提取**：从第二个MASK位置提取句子表示
- **MLP层**：可选的MLP投影层

### 2. 训练流程
1. **数据准备**：使用wiki1m_for_simcse.txt作为训练数据
2. **模板应用**：为每个句子应用多个模板（锚句、正例、负例）
3. **表示提取**：从MASK位置提取句子表示
4. **去噪处理**：使用Delta机制去除位置噪声
5. **对比学习**：使用InfoNCE损失进行训练

### 3. 评估流程
1. **模型加载**：加载训练好的模型
2. **SentEval评估**：在SentEval基准测试上评估
3. **任务类型**：
   - STS任务：语义文本相似度
   - 迁移任务：分类和推理任务

## 使用方法

### 环境要求

- Python 3.9+
- PyTorch 2.6.0+
- Transformers 4.28.1+
- 其他依赖见 `requirement_cuda_python390.txt`

### 安装依赖

```bash
pip install -r requirement_cuda_python390.txt
```

### 训练

#### Linux CUDA平台
```bash
# 使用默认配置
bash train_linux_cuda.sh

# 或使用自定义配置文件
python cot_bert_train.py configs/train_linux_cuda.yaml
```

#### Mac M4平台
```bash
# 使用默认配置
bash train_mac_m4.sh

# 或使用自定义配置文件
python cot_bert_train.py configs/train_mac_m4.yaml
```

#### 通用训练
```bash
# 自动检测平台并使用对应配置
bash train.sh

# 或手动指定配置文件
python cot_bert_train.py configs/train_default.yaml
```

### 评估

#### Linux CUDA平台
```bash
# 使用默认配置
bash evaluate_linux_cuda.sh

# 或使用自定义配置文件
python cot_bert_evaluation.py configs/evaluation_linux_cuda.yaml
```

#### Mac M4平台
```bash
# 使用默认配置
bash evaluate_mac_m4.sh

# 或使用自定义配置文件
python cot_bert_evaluation.py configs/evaluation_mac_m4.yaml
```

#### 通用评估
```bash
# 自动检测平台并使用对应配置
bash evaluation.sh

# 或手动指定配置文件
python cot_bert_evaluation.py configs/evaluation_default.yaml
```

## 配置说明

### 训练配置（configs/train_*.yaml）

主要参数：
- `model_name_or_path`: 预训练模型路径（默认：bert-base-uncased）
- `train_file`: 训练数据文件路径（默认：../data/wiki1m_for_simcse.txt）
- `output_dir`: 模型输出目录（默认：../result/CoT-Bert）
- `per_device_train_batch_size`: 每设备批次大小
  - Linux CUDA: 128
  - Mac M4: 32
- `learning_rate`: 学习率（默认：1e-5）
- `num_train_epochs`: 训练轮数（默认：1）
- `temp`: InfoNCE温度参数（默认：0.05）
- `mask_num`: MASK数量（默认：2）
- `mask_embedding_sentence_template`: 锚句模板
- `mask_embedding_sentence_different_template`: 正例模板
- `mask_embedding_sentence_negative_template`: 负例模板
- `mask_embedding_sentence_delta`: 是否使用Delta去噪（默认：true）

### 评估配置（configs/evaluation_*.yaml）

主要参数：
- `model_name_or_path`: 训练好的模型路径（默认：../result/CoT-Bert）
- `mode`: 评估模式（test/dev/fasttest）
- `task_set`: 任务集合（sts/transfer/full）
- `mask_embedding_sentence`: 是否使用MASK嵌入（默认：true）
- `mask_num`: MASK数量（默认：2）
- `mask_embedding_sentence_template`: 模板格式
- `mask_embedding_sentence_delta`: 是否使用Delta去噪（默认：true）

## 平台支持

### Mac M4平台
- **设备**：Apple Silicon (M4)
- **加速**：MPS (Metal Performance Shaders)
- **批次大小**：32（受内存限制）
- **工作进程**：4

### Linux CUDA平台
- **设备**：NVIDIA GPU
- **加速**：CUDA
- **批次大小**：128
- **工作进程**：16

### 自动平台检测
项目会自动检测运行平台并选择相应的配置文件：
- Mac M4 → `configs/train_mac_m4.yaml`
- Linux CUDA → `configs/train_linux_cuda.yaml`
- 其他平台 → `configs/train_default.yaml`

## 实验数据集

### 训练数据
- **wiki1m_for_simcse.txt**: 从Wikipedia提取的100万条句子对

### 评估数据（SentEval）
- **STS任务**：
  - STS12, STS13, STS14, STS15, STS16
  - STSBenchmark
  - SICKRelatedness
- **迁移任务**：
  - MR: 电影评论情感分析
  - CR: 客户评论情感分析
  - SUBJ: 主观性分类
  - MPQA: 观点极性分类
  - SST2: 斯坦福情感树库
  - TREC: 问题分类
  - MRPC: 微软研究段落语料库

## 技术特点

### 1. 跨平台支持
- 自动检测运行平台（Mac M4 / Linux CUDA）
- 根据平台自动调整批次大小和工作进程数
- 统一的配置文件接口

### 2. 模块化设计
- 平台工具模块（platform_utils.py）
- 参数解析工具（parse_args_util.py）
- 日志工具（lmf_log_util.py）

### 3. 灵活的配置
- YAML配置文件支持
- 支持自定义配置文件覆盖默认配置
- 命令行参数支持

## 预期结果

### 性能指标
- **STS-B**: Spearman相关系数 > 85%
- **SICK-R**: Spearman相关系数 > 80%
- **MRPC**: F1分数 > 90%

### 效率指标
- **训练时间**：单GPU约2-4小时（取决于硬件）
- **推理时间**：单条句子 < 10ms
- **内存使用**：约8-16GB（取决于批次大小）

## 注意事项

1. **数据路径**：确保训练数据文件路径正确（默认：`../data/wiki1m_for_simcse.txt`）
2. **模型路径**：确保预训练模型路径正确（默认：`bert-base-uncased`）
3. **输出目录**：确保输出目录有写入权限（默认：`../result/CoT-Bert`）
4. **平台配置**：根据运行平台选择合适的配置文件
5. **CUDA设置**：Linux平台会自动设置CUDA_VISIBLE_DEVICES，无需手动设置

## 参考文献

1. Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.
2. Jiang, T., et al. (2022). PromptBERT: Improving BERT Sentence Embeddings with Prompts. ACL.
3. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.

## 许可证

本项目采用MIT许可证。

