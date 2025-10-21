#!/usr/bin/env python3
"""
SPR模型测试脚本
验证模型的基本功能是否正常
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spr_model import BertForSPR, ProjectionNetwork, PredictionNetwork
from transformers import BertConfig, BertTokenizer

def test_projection_network():
    """测试投影网络"""
    print("测试ProjectionNetwork...")
    config = BertConfig()
    proj_net = ProjectionNetwork(config)
    
    # 创建测试输入
    batch_size = 2
    hidden_size = config.hidden_size
    test_input = torch.randn(batch_size, hidden_size)
    
    # 前向传播
    output = proj_net(test_input)
    
    assert output.shape == test_input.shape, f"输出形状不匹配: {output.shape} vs {test_input.shape}"
    print("✓ ProjectionNetwork测试通过")

def test_prediction_network():
    """测试预测网络"""
    print("测试PredictionNetwork...")
    config = BertConfig()
    pred_net = PredictionNetwork(config)
    
    # 创建测试输入
    batch_size = 2
    hidden_size = config.hidden_size
    h1 = torch.randn(batch_size, hidden_size)
    h2 = torch.randn(batch_size, hidden_size)
    
    # 前向传播
    output = pred_net(h1, h2)
    
    assert output.shape == (batch_size, hidden_size), f"输出形状不匹配: {output.shape}"
    print("✓ PredictionNetwork测试通过")

def test_bert_for_spr():
    """测试BertForSPR模型"""
    print("测试BertForSPR...")
    
    # 创建模拟的model_args
    class MockModelArgs:
        def __init__(self):
            self.spr_denoising = True
    
    model_args = MockModelArgs()
    
    # 创建配置和模型
    config = BertConfig()
    model = BertForSPR(config, model_args=model_args)
    
    # 创建测试输入
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # 前向传播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    assert hasattr(outputs, 'loss'), "输出应该包含loss"
    assert hasattr(outputs, 'logits'), "输出应该包含logits"
    print("✓ BertForSPR测试通过")

def main():
    """主测试函数"""
    print("开始SPR模型测试...")
    print("=" * 50)
    
    try:
        test_projection_network()
        test_prediction_network()
        test_bert_for_spr()
        
        print("=" * 50)
        print("🎉 所有测试通过！SPR模型实现正确。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
