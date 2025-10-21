# SPR 跨平台训练指南

## 概述

SPR (Self-Projection Regularization) 项目现在支持跨平台训练：
- **Mac M4**: 用于调试和开发
- **Linux NVIDIA CUDA**: 用于大规模训练

## 平台支持

### Mac M4 (Apple Silicon)
- **设备**: MPS (Metal Performance Shaders)
- **批次大小**: 8 (内存限制)
- **梯度累积**: 8 (保持有效批次大小64)
- **工作进程**: 4
- **FP16**: 不支持

### Linux NVIDIA CUDA
- **设备**: CUDA GPU
- **批次大小**: 64
- **梯度累积**: 2
- **工作进程**: 16
- **FP16**: 支持混合精度训练

## 文件结构

```
code-for-SPR/
├── platform_utils.py              # 跨平台工具模块
├── configs/
│   ├── train_mac_m4.yaml          # Mac M4 训练配置
│   ├── train_linux_cuda.yaml      # Linux CUDA 训练配置
│   ├── train_default.yaml         # 默认配置
│   └── evaluation_default.yaml     # 评估配置
├── train_mac_m4.sh                # Mac M4 训练脚本
├── train_linux_cuda.sh            # Linux CUDA 训练脚本
├── evaluation_mac_m4.sh           # Mac M4 评估脚本
├── evaluation_linux_cuda.sh       # Linux CUDA 评估脚本
├── spr_train.py                   # 训练主程序（已修改）
├── spr_evaluation.py              # 评估程序（已修改）
└── spr_model.py                   # 模型定义
```

## 使用方法

### Mac M4 调试训练

```bash
# 激活虚拟环境
cd /Users/lmf/Documents/local/code/CoT-BERT
source myvenv3.8/bin/activate

# 进入SPR目录
cd code-for-SPR

# 运行Mac M4训练
./train_mac_m4.sh
```

### Linux NVIDIA CUDA 训练

```bash
# 激活虚拟环境
source myvenv3.8/bin/activate

# 进入SPR目录
cd code-for-SPR

# 运行Linux CUDA训练
./train_linux_cuda.sh
```

### 手动运行（自动检测平台）

```bash
# 训练
python spr_train.py

# 评估
python spr_evaluation.py configs/evaluation_default.yaml
```

## 配置说明

### Mac M4 配置特点
- `per_device_train_batch_size: 8` - 小批次避免内存溢出
- `gradient_accumulation_steps: 8` - 保持有效批次大小
- `use_mps_device: true` - 启用MPS加速
- `fp16: false` - 禁用混合精度
- `preprocessing_num_workers: 4` - 减少工作进程

### Linux CUDA 配置特点
- `per_device_train_batch_size: 64` - 大批次充分利用GPU
- `gradient_accumulation_steps: 2` - 适中的梯度累积
- `fp16: true` - 启用混合精度训练
- `preprocessing_num_workers: 16` - 更多工作进程
- `dataloader_num_workers: 8` - 数据加载优化

## 环境变量

### Mac M4
```bash
export MPS_FALLBACK=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Linux CUDA
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据需要调整
export NCCL_DEBUG=INFO  # 分布式训练调试
```

## 性能对比

| 平台 | 批次大小 | 内存使用 | 训练速度 | 适用场景 |
|------|----------|----------|----------|----------|
| Mac M4 | 8 | ~8GB | 中等 | 调试、开发 |
| Linux CUDA | 64 | ~16GB | 快 | 生产训练 |

## 故障排除

### Mac M4 常见问题

1. **MPS不可用**
   ```bash
   # 检查MPS支持
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

2. **内存不足**
   - 减小批次大小到4或更小
   - 增加梯度累积步数

3. **MPS回退到CPU**
   - 设置环境变量 `MPS_FALLBACK=1`
   - 检查PyTorch版本 >= 1.12

### Linux CUDA 常见问题

1. **CUDA不可用**
   ```bash
   # 检查CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **GPU内存不足**
   - 减小批次大小
   - 启用梯度检查点
   - 使用混合精度训练

3. **多GPU训练问题**
   - 检查NCCL安装
   - 设置正确的CUDA_VISIBLE_DEVICES

## 开发工作流

### 推荐工作流
1. **Mac M4**: 代码开发、调试、小规模测试
2. **Linux CUDA**: 大规模训练、最终实验

### 代码同步
- 使用Git同步代码
- 确保配置文件路径正确
- 检查数据文件路径

## 注意事项

1. **数据路径**: 确保数据文件在不同平台上路径正确
2. **模型路径**: 预训练模型路径需要调整
3. **输出路径**: 结果保存路径需要检查
4. **依赖版本**: 确保PyTorch版本兼容

## 技术支持

如遇到问题，请检查：
1. 平台检测是否正确
2. 配置文件是否匹配
3. 环境变量是否设置
4. 依赖包是否安装完整
