
# SPR (Self-Projection Regularization) 实验工程
## 我的提示词
为了发表sci论文获取实验数据，我需要设计一个实验，创建一个新的python工程目录，参考code-bert-LMF项目代码，训练具体步骤：
1.  设计prompt ”The sentence of [X] means [mask], and also means [mask]",其中X为实际句子占位符，将prompt 输入到模型pretrain_models/bert-base-uncased进行编码，得到句子X正编码表示h1*和编码表示h2*， h1*,h2*可理解为句子X语义的不同方面, 可解决为一束白光经过三菱镜后分解成2束不同颜色的光类似. 然后h1*，h2*去掉prompt模版本身的噪声，得到h1和h2.

2. 然后采用自正则化（Self-Projection Regularization, SPR）的投影-预测框架（Projection-Prediction Consistency），分别设计h1 和h2对应的投影网络project1-network，project2-network，得到h1-project, h2-project，类似于h1、h2两束光分别经过另外两个三菱镜即这里的project1-network，project2-network。

3. 分别设计(h1, h2)和（h1-project, h2-project）两对Prediction 网络框架，分别为Prediction-network， Prediction-network-project，将(h1, h2)输入到Prediction-network预测网络，(h1-project, h2-project）输入到Prediction-network-project网络，得到投影前后两个预测值：h_prediction, h_predication_project

4. 训练目标loss=||norm(h_prediction) - norm(h_predication_project)||

验证步骤：
采用上面训练步骤3中的h_prediction为预测值


## 项目概述

本项目实现了基于自投影正则化（SPR）的语义表示学习实验，用于发表SCI论文。SPR通过prompt模板将句子编码为两个语义方面，然后使用投影网络和预测网络计算投影前后的一致性损失。

## 核心思想

1. **Prompt编码**: 使用"The sentence of [X] means [mask], and also means [mask]"将句子编码为两个语义方面h1*, h2*
2. **去噪**: 减去prompt模板本身的噪声，得到干净的h1, h2
3. **投影**: 通过两个独立的投影网络得到h1-project, h2-project
4. **预测**: 使用两个预测网络分别处理(h1,h2)和(h1-project,h2-project)
5. **损失**: 计算投影前后预测的一致性损失

## 文件结构

```
code-for-SPR/
├── spr_model.py              # 模型定义
├── spr_train.py              # 训练主程序
├── spr_trainer.py            # 自定义训练器
├── spr_evaluation.py         # 评估程序
├── parse_args_util.py        # 配置文件解析工具
├── lmf_log_util.py          # 日志工具
├── configs/
│   ├── train_default.yaml    # 训练配置
│   └── evaluation_default.yaml # 评估配置
├── train.sh                  # 训练脚本
└── evaluation.sh            # 评估脚本
```

## 使用方法

### 训练

```bash
cd code-for-SPR
./train.sh
```

或者直接运行：

```bash
python spr_train.py
```

### 评估

```bash
cd code-for-SPR
./evaluation.sh
```

或者直接运行：

```bash
python spr_evaluation.py
```

## 配置说明

### 训练配置 (configs/train_default.yaml)

- `model_name_or_path`: BERT预训练模型路径
- `train_file`: 训练数据文件
- `prompt_template`: SPR prompt模板
- `mask_num`: mask token数量（固定为2）
- `spr_denoising`: 是否使用去噪

### 评估配置 (configs/evaluation_default.yaml)

- `model_name_or_path`: 训练好的模型路径
- `mode`: 评估模式（dev/test/fasttest）
- `task_set`: 评估任务集（sts/transfer/full）

## 实验流程

1. **训练阶段**:
   - 加载wiki1m数据集
   - 应用prompt模板编码句子
   - 提取两个mask位置的hidden states
   - 计算去噪、投影、预测
   - 优化SPR一致性损失

2. **评估阶段**:
   - 使用训练好的模型
   - 对输入句子应用prompt编码
   - 使用h_prediction作为句子表示
   - 在SentEval的STS任务上评估

## 关键特性

- **自投影正则化**: 通过投影前后预测的一致性约束学习更好的语义表示
- **双方面语义**: 将句子分解为两个语义方面进行建模
- **去噪机制**: 去除prompt模板本身的噪声影响
- **SentEval评估**: 在标准语义相似度任务上评估性能

## 依赖环境

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- SentEval
- NumPy

## 注意事项

1. 确保预训练模型路径正确
2. 训练数据文件路径需要存在
3. SentEval数据路径需要正确配置
4. 根据GPU内存调整batch size
