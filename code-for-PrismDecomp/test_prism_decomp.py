#!/usr/bin/env python3
"""
PrismDecomp模型测试脚本
测试基于棱镜分解的多语义句子表示学习模型的基本功能
"""

import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# 添加项目路径
sys.path.append('..')

from prism_decomp_model import BertForPrismDecomp, SemanticDecomposer, MultiSemanticSPR, SPR_Module

def test_semantic_decomposer():
    """测试语义分解器"""
    print("测试语义分解器...")
    
    hidden_dim = 768
    num_semantics = 7
    batch_size = 4
    
    decomposer = SemanticDecomposer(hidden_dim, num_semantics)
    
    # 创建测试输入
    sentence_repr = torch.randn(batch_size, hidden_dim)
    
    # 前向传播
    semantic_reprs, orth_loss = decomposer(sentence_repr)
    
    print(f"输入形状: {sentence_repr.shape}")
    print(f"输出形状: {semantic_reprs.shape}")
    print(f"语义维度数量: {semantic_reprs.shape[1]}")
    print(f"软正交损失: {orth_loss.item():.4f}")
    
    # 验证输出形状
    assert semantic_reprs.shape == (batch_size, num_semantics, hidden_dim), f"期望形状 {(batch_size, num_semantics, hidden_dim)}, 实际形状 {semantic_reprs.shape}"
    assert orth_loss.item() >= 0, "软正交损失应该非负"
    
    print("✅ 语义分解器测试通过!")
    return True

def test_spr_module():
    """测试SPR模块"""
    print("测试SPR模块...")
    
    hidden_dim = 768
    batch_size = 4
    
    spr_module = SPR_Module(hidden_dim)
    
    # 创建测试输入
    semantic_repr = torch.randn(batch_size, hidden_dim)
    
    # 前向传播
    processed_repr, spr_loss = spr_module(semantic_repr)
    
    print(f"输入形状: {semantic_repr.shape}")
    print(f"输出形状: {processed_repr.shape}")
    print(f"SPR损失值: {spr_loss.item():.4f}")
    
    # 验证输出形状
    assert processed_repr.shape == (batch_size, hidden_dim), f"期望形状 {(batch_size, hidden_dim)}, 实际形状 {processed_repr.shape}"
    assert spr_loss.item() > 0, "SPR损失应该大于0"
    
    print("✅ SPR模块测试通过!")
    return True

def test_multisemantic_spr():
    """测试多语义SPR模型"""
    print("测试多语义SPR模型...")
    
    hidden_dim = 768
    num_semantics = 7
    batch_size = 4
    
    multisemantic_spr = MultiSemanticSPR(hidden_dim, num_semantics, lambda1=1.0, lambda2=0.01)
    
    # 创建测试输入
    sentence_repr = torch.randn(batch_size, hidden_dim)
    
    # 前向传播
    final_repr, total_loss, semantic_spr_losses = multisemantic_spr(sentence_repr)
    
    print(f"输入形状: {sentence_repr.shape}")
    print(f"输出形状: {final_repr.shape}")
    print(f"总损失: {total_loss.item():.4f}")
    print(f"语义SPR损失数量: {len(semantic_spr_losses)}")
    print(f"λ₁ (自正则化权重): {multisemantic_spr.lambda1}")
    print(f"λ₂ (软正交权重): {multisemantic_spr.lambda2}")
    
    # 验证输出形状
    assert final_repr.shape == (batch_size, hidden_dim), f"期望形状 {(batch_size, hidden_dim)}, 实际形状 {final_repr.shape}"
    assert len(semantic_spr_losses) == num_semantics, f"期望语义损失数量 {num_semantics}, 实际数量 {len(semantic_spr_losses)}"
    assert total_loss.item() > 0, "总损失应该大于0"
    
    print("✅ 多语义SPR模型测试通过!")
    return True

def test_bert_for_prism_decomp():
    """测试完整的BERT PrismDecomp模型"""
    print("测试BERT PrismDecomp模型...")
    
    try:
        # 使用一个简单的配置进行测试
        config = AutoConfig.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # 创建模型参数
        class ModelArgs:
            def __init__(self):
                self.num_semantics = 7
                self.orthogonal_constraint = True
                self.semantic_weights_learnable = True
        
        model_args = ModelArgs()
        
        # 创建模型
        model = BertForPrismDecomp(config, model_args=model_args)
        
        # 创建测试输入（双句用于对比学习）
        batch_size = 2
        num_sent = 2  # 每个样本包含两个句子
        seq_length = 32
        
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, num_sent, seq_length))
        attention_mask = torch.ones(batch_size, num_sent, seq_length)
        
        print(f"输入形状: {input_ids.shape}")
        print(f"注意力掩码形状: {attention_mask.shape}")
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        print(f"输出logits形状: {outputs.logits.shape}")
        print(f"总损失值: {outputs.loss.item():.4f}")
        
        # 验证输出形状
        assert outputs.logits.shape == (batch_size, config.hidden_size), f"期望形状 {(batch_size, config.hidden_size)}, 实际形状 {outputs.logits.shape}"
        assert outputs.loss.item() > 0, "总损失应该大于0"
        
        print("✅ BERT PrismDecomp模型测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ BERT PrismDecomp模型测试失败: {e}")
        return False

def test_sentence_embedding():
    """测试句子嵌入功能"""
    print("测试句子嵌入功能...")
    
    try:
        # 使用一个简单的配置进行测试
        config = AutoConfig.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # 创建模型参数
        class ModelArgs:
            def __init__(self):
                self.num_semantics = 7
                self.orthogonal_constraint = True
                self.semantic_weights_learnable = True
        
        model_args = ModelArgs()
        
        # 创建模型
        model = BertForPrismDecomp(config, model_args=model_args)
        
        # 创建测试输入
        sentences = ["This is a test sentence.", "Another test sentence for evaluation."]
        
        # 编码句子
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        
        print(f"输入句子数量: {len(sentences)}")
        print(f"输入形状: {inputs['input_ids'].shape}")
        
        # 获取句子嵌入
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                sent_emb=True,
                return_dict=True
            )
        
        print(f"句子嵌入形状: {outputs.pooler_output.shape}")
        
        # 验证输出形状
        assert outputs.pooler_output.shape[0] == len(sentences), f"期望句子数量 {len(sentences)}, 实际数量 {outputs.pooler_output.shape[0]}"
        assert outputs.pooler_output.shape[1] == config.hidden_size, f"期望隐藏维度 {config.hidden_size}, 实际维度 {outputs.pooler_output.shape[1]}"
        
        print("✅ 句子嵌入功能测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 句子嵌入功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("PrismDecomp模型功能测试")
    print("=" * 60)
    
    tests = [
        test_semantic_decomposer,
        test_spr_module,
        test_multisemantic_spr,
        test_bert_for_prism_decomp,
        test_sentence_embedding,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 失败: {e}")
            print()
    
    print("=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("🎉 所有测试通过! PrismDecomp模型实现正确!")
        return True
    else:
        print("⚠️  部分测试失败，请检查实现!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
