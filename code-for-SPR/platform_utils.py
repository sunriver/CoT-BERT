"""
跨平台工具模块
支持Mac M4调试和Linux NVIDIA CUDA训练
"""

import os
import platform
import torch
import logging

logger = logging.getLogger(__name__)

def detect_platform():
    """检测当前运行平台"""
    system = platform.system()
    machine = platform.machine()
    
    if system == "Darwin" and machine == "arm64":
        return "mac_m4"
    elif system == "Linux":
        return "linux_cuda"
    elif system == "Darwin":
        return "mac_intel"
    elif system == "Windows":
        return "windows"
    else:
        return "unknown"

def setup_device_config():
    """根据平台设置设备配置"""
    platform_type = detect_platform()
    
    config = {
        "platform": platform_type,
        "device": "cpu",
        "use_mps": False,
        "use_cuda": False,
        "fp16": False,
        "batch_size": 8,
        "gradient_accumulation": 8,
        "num_workers": 4,
        "cuda_devices": None
    }
    
    if platform_type == "mac_m4":
        if torch.backends.mps.is_available():
            config.update({
                "device": "mps",
                "use_mps": True,
                "batch_size": 8,
                "gradient_accumulation": 8,
                "num_workers": 4
            })
            logger.info("Mac M4 平台：使用 MPS 设备")
        else:
            logger.info("Mac M4 平台：使用 CPU 设备")
            
    elif platform_type == "linux_cuda":
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            config.update({
                "device": "cuda",
                "use_cuda": True,
                "fp16": True,
                "batch_size": 64,
                "gradient_accumulation": 2,
                "num_workers": 16,
                "cuda_devices": list(range(gpu_count))
            })
            logger.info(f"Linux CUDA 平台：使用 {gpu_count} 个GPU")
        else:
            logger.info("Linux 平台：使用 CPU 设备")
            
    else:
        logger.info(f"{platform_type} 平台：使用 CPU 设备")
    
    return config

def get_platform_config_file():
    """根据平台返回对应的配置文件"""
    platform_type = detect_platform()
    
    if platform_type == "mac_m4":
        return "configs/train_mac_m4.yaml"
    elif platform_type == "linux_cuda":
        return "configs/train_linux_cuda.yaml"
    else:
        return "configs/train_default.yaml"

def setup_cuda_environment():
    """设置CUDA环境变量"""
    platform_type = detect_platform()
    
    if platform_type == "linux_cuda" and torch.cuda.is_available():
        # 设置CUDA设备
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 4:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        elif gpu_count >= 2:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        logger.info(f"设置CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

def get_optimal_batch_size(base_batch_size=64):
    """根据平台获取最优批次大小"""
    platform_type = detect_platform()
    
    if platform_type == "mac_m4":
        return min(base_batch_size // 8, 8)  # Mac M4内存限制
    elif platform_type == "linux_cuda":
        return base_batch_size
    else:
        return min(base_batch_size // 4, 16)  # CPU训练

def get_optimal_workers():
    """根据平台获取最优工作进程数"""
    platform_type = detect_platform()
    
    if platform_type == "mac_m4":
        return 4  # Mac M4 CPU核心数限制
    elif platform_type == "linux_cuda":
        return 16  # Linux服务器通常有更多核心
    else:
        return 8

def print_platform_info():
    """打印平台信息"""
    platform_type = detect_platform()
    config = setup_device_config()
    
    print("=" * 50)
    print("平台信息")
    print("=" * 50)
    print(f"操作系统: {platform.system()}")
    print(f"架构: {platform.machine()}")
    print(f"平台类型: {platform_type}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {config['device']}")
    print(f"使用MPS: {config['use_mps']}")
    print(f"使用CUDA: {config['use_cuda']}")
    print(f"FP16支持: {config['fp16']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"梯度累积: {config['gradient_accumulation']}")
    print(f"工作进程: {config['num_workers']}")
    
    if config['use_cuda'] and torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("=" * 50)

if __name__ == "__main__":
    print_platform_info()
