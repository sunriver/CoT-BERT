## CrossTemplateCoT

本项目实现了 Cross-Template Chain-of-Thought (CrossTemplateCoT) 句子表示学习框架，位于 `CoT-BERT/code-for-CrossTemplateCoT`。

### 核心思路
- 使用 3 个 CoT 模板（锚句、正样本、负样本）：
  1. 锚句：`The sentence of "[X]" means [MASK], so it can be summarized as [MASK].`
  2. 正样本：`The sentence ："[X]" means [MASK], so it can be summarized as [MASK].`
  3. 负样本：`The sentence ："[X]" does not mean [MASK], so it cannot be summarized as [MASK]`
- 每个模板包含 2 个 `[MASK]`，共提取 6 个 MASK 表示。
- 使用 CoT-BERT 一致的 delta 去噪方法：用与句子长度相同的 PAD 替换 `[X]`，提取位置噪声并从真实 MASK 表示中减去。
- 损失函数：
  - **L1**：第一个 MASK 位置的 InfoNCE（锚句 vs 正样本，负样本来自批内所有样本）。
  - **L2**：第二个 MASK 位置的 InfoNCE。
  - **L3**：约束项  
    \\[
    L3 = \\frac{\\lVert h^{(2)}_{pos} - h^{(1)}_{anc} \\rVert_2 + \\lVert h^{(2)}_{anc} - h^{(1)}_{pos} \\rVert_2}{\\lVert h^{(2)}_{anc} \\rVert_2 + \\lVert h^{(2)}_{pos} \\rVert_2^2 + \\varepsilon}
    \\]
  - 总损失：`Total Loss = L1 + L2 + L3`（支持通过配置调节各自权重）。

### 主要文件
- `cross_template_cot_model.py`：BERT 主模型，包含：
  - `BertForCrossTemplateCoT` 类
  - `cross_template_cot_forward`：训练前向，计算 L1/L2/L3。
  - `cross_template_cot_sentemb_forward`：评估前向，输出句子向量（用于 SentEval）。
- `cross_template_cot_trainer.py`：
  - `CrossTemplateCoTTrainer` 继承自 `Trainer`，实现：
    - `compute_loss`：组装模板并调用模型。
    - `evaluate`：内置 SentEval 评估。
    - `train` 与 `_save_checkpoint`：参考 TwoStageCoT，全流程训练与最优 checkpoint 保存。
- `cross_template_cot_train.py`：
  - 定义 `ModelArguments` / `DataTrainingArguments` / `OurTrainingArguments`。
  - `prepare_features`：对每个句子构造 3 个模板输入 (anchor / positive / negative)。
  - 主入口 `main()`：加载数据、模型，构建 `CrossTemplateCoTTrainer` 并启动训练。
- `cross_template_cot_evaluation.py`：
  - 读取评估配置，加载训练好的 CrossTemplateCoT 模型。
  - 使用 SentEval 在 STS/Transfer 任务上评估句子表示。

### 配置文件
位于 `configs/`：
- 训练：
  - `train_default.yaml`
  - `train_mac_m4.yaml`
  - `train_linux_cuda.yaml`
- 评估：
  - `evaluation_default.yaml`
  - `evaluation_mac_m4.yaml`
  - `evaluation_linux_cuda.yaml`

所有配置均包含：
- HuggingFace 训练参数（batch size, learning rate, eval_steps, save_steps 等）。
- 模板定义：
  - `mask_embedding_sentence_template`
  - `mask_embedding_sentence_different_template`
  - `mask_embedding_sentence_negative_template`
- 去噪与损失权重设置：
  - `mask_embedding_sentence_delta`
  - `mask_embedding_sentence_delta_freeze`
  - `mask_embedding_sentence_org_mlp`
  - `process_supervision_weight_1`
  - `process_supervision_weight_2`
  - `constraint_weight`

### Shell 脚本
- 训练：
  - `train.sh`：默认配置训练。
  - `train_mac_m4.sh`：Mac M4 本地调试。
  - `train_linux_cuda.sh`：Linux CUDA 环境训练。
- 评估：
  - `evaluation.sh`：默认配置评估。
  - `evaluate_mac_m4.sh`：Mac M4 评估。
  - `evaluate_linux_cuda.sh`：Linux CUDA 评估。

脚本默认会：
- 激活上级目录下的 `../.venv` 虚拟环境（Linux CUDA 版本留有注释，可按需启用）。
- 设置 `PYTHONPATH`，保证本项目与 `SentEval` 可被正确导入。

### 使用示例
在 `CoT-BERT/code-for-CrossTemplateCoT` 目录下：
- Mac 本地训练：
  ```bash
  bash train_mac_m4.sh
  ```
- Linux CUDA 训练：
  ```bash
  bash train_linux_cuda.sh
  ```
- 训练完成后运行 SentEval 评估：
  ```bash
  bash evaluation.sh          # 使用默认模型
  bash evaluate_mac_m4.sh     # Mac M4 模型
  bash evaluate_linux_cuda.sh # Linux CUDA 模型
  ```


