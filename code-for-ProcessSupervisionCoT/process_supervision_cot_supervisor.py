"""
过程监督评估器模块
用于评估 CoT 模板中 mask1 和 mask2 的推理路径质量
"""

import re
import torch
from typing import List, Dict, Optional, Tuple
from openai import OpenAI


# 评估 prompt 模板（参考 LongRePS 的实现，适配 CoT-BERT）
EVAL_PROMPT_MASK1 = '''[Sentence]
{sentence}

[The Start of Mask1 Reasoning Path]
{reasoning}
[The End of Mask1 Reasoning Path]

[System]
We would like to request your feedback on the quality of the reasoning process for the first mask token in the Chain-of-Thought template. 
The model receives a sentence and generates a reasoning path to understand what the sentence means. Above, we have provided both the sentence and the model's reasoning process for the first mask token.

Please assess the model's reasoning process based on the following aspects:

1. Logical Coherence:
- The model should break down the sentence appropriately
- The use of information should follow logical patterns
- The chain of reasoning should be sound

2. Completeness:
- The reasoning process should capture the essential meaning of the sentence
- The model should not miss important information

3. Conciseness:
- Only information relevant to understanding the sentence should be included
- The model should avoid listing excessive or irrelevant information

Please rate whether this reasoning path is suitable for the sentence. The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
Please note that if the assistant's reasoning process fully meets the above criteria, its overall rating should be full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS ON A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
<start output>
Evaluation evidence: your evaluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:'''

EVAL_PROMPT_MASK2 = '''[Sentence]
{sentence}

[The Start of Mask2 Reasoning Path]
{reasoning}
[The End of Mask2 Reasoning Path]

[System]
We would like to request your feedback on the quality of the reasoning process for the second mask token in the Chain-of-Thought template. 
The model receives a sentence and generates a reasoning path to summarize the sentence. Above, we have provided both the sentence and the model's reasoning process for the second mask token.

Please assess the model's reasoning process based on the following aspects:

1. Logical Coherence:
- The model should appropriately summarize the sentence
- The summary should follow from the first mask's reasoning
- The chain of reasoning should be sound

2. Completeness:
- The reasoning process should capture the essential summary of the sentence
- The model should not miss important information

3. Conciseness:
- Only information relevant to summarizing the sentence should be included
- The model should avoid listing excessive or irrelevant information

Please rate whether this reasoning path is suitable for the sentence. The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
Please note that if the assistant's reasoning process fully meets the above criteria, its overall rating should be full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS ON A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
<start output>
Evaluation evidence: your evaluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:'''


class ProcessSupervisor:
    """
    过程监督评估器类
    用于评估 mask1 和 mask2 的推理路径质量
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        """
        初始化过程监督评估器
        Args:
            api_key: OpenAI API key（如果为 None，则从环境变量获取）
            model: 使用的 LLM 模型名称
            base_url: API base URL（可选，用于自定义 API 端点）
        """
        if api_key is None:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("需要提供 api_key 或设置 OPENAI_API_KEY 环境变量")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.eval_prompt_mask1 = EVAL_PROMPT_MASK1
        self.eval_prompt_mask2 = EVAL_PROMPT_MASK2
    
    def evaluate_mask1(self, sentence: str, reasoning_path: str) -> float:
        """
        评估 mask1 的推理路径质量
        Args:
            sentence: 输入句子
            reasoning_path: mask1 的推理路径文本
        Returns:
            score: 评分（1-100）
        """
        prompt = self.eval_prompt_mask1.format(
            sentence=sentence,
            reasoning=reasoning_path
        )
        score = self._get_score(prompt)
        return score
    
    def evaluate_mask2(self, sentence: str, reasoning_path: str) -> float:
        """
        评估 mask2 的推理路径质量
        Args:
            sentence: 输入句子
            reasoning_path: mask2 的推理路径文本
        Returns:
            score: 评分（1-100）
        """
        prompt = self.eval_prompt_mask2.format(
            sentence=sentence,
            reasoning=reasoning_path
        )
        score = self._get_score(prompt)
        return score
    
    def _get_score(self, prompt: str) -> float:
        """
        调用 LLM API 获取评分
        Args:
            prompt: 评估 prompt
        Returns:
            score: 评分（1-100）
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            
            content = response.choices[0].message.content
            
            # 解析评分
            score = self._parse_score(content)
            return score
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            # 返回默认评分（中等质量）
            return 50.0
    
    def _parse_score(self, content: str) -> float:
        """
        从 LLM 响应中解析评分
        Args:
            content: LLM 响应文本
        Returns:
            score: 评分（1-100）
        """
        # 尝试匹配 [[score]] 格式
        pattern = r'\[\[(\d+)\]\]'
        match = re.search(pattern, content)
        if match:
            score = float(match.group(1))
            # 确保评分在 1-100 范围内
            score = max(1.0, min(100.0, score))
            return score
        
        # 如果找不到 [[score]] 格式，尝试匹配 "Rating: score" 格式
        pattern = r'Rating:\s*(\d+)'
        match = re.search(pattern, content)
        if match:
            score = float(match.group(1))
            score = max(1.0, min(100.0, score))
            return score
        
        # 如果都找不到，返回默认评分
        print(f"Warning: Could not parse score from content: {content[:200]}")
        return 50.0


def extract_reasoning_paths(input_ids: torch.Tensor, 
                            mask_positions: torch.Tensor,
                            tokenizer,
                            window_size: int = 10) -> List[Dict[str, str]]:
    """
    从输入中提取 mask1 和 mask2 位置的推理路径
    Args:
        input_ids: [batch_size, seq_len] 输入 token ids
        mask_positions: [batch_size, 2] mask1 和 mask2 的位置索引
        tokenizer: tokenizer 对象
        window_size: 上下文窗口大小
    Returns:
        reasoning_paths: 推理路径列表，每个元素包含 'mask1_path' 和 'mask2_path'
    """
    reasoning_paths = []
    batch_size = input_ids.size(0)
    
    for i in range(batch_size):
        mask1_pos = mask_positions[i, 0].item()
        mask2_pos = mask_positions[i, 1].item()
        
        # 提取 mask1 周围的上下文
        start1 = max(0, mask1_pos - window_size)
        end1 = min(input_ids.size(1), mask1_pos + window_size + 1)
        context1 = input_ids[i, start1:end1]
        mask1_path = tokenizer.decode(context1, skip_special_tokens=False)
        
        # 提取 mask2 周围的上下文
        start2 = max(0, mask2_pos - window_size)
        end2 = min(input_ids.size(1), mask2_pos + window_size + 1)
        context2 = input_ids[i, start2:end2]
        mask2_path = tokenizer.decode(context2, skip_special_tokens=False)
        
        reasoning_paths.append({
            'mask1_path': mask1_path,
            'mask2_path': mask2_path
        })
    
    return reasoning_paths


def compute_process_supervision_loss(mask1_scores: torch.Tensor,
                                    mask2_scores: torch.Tensor,
                                    device: torch.device) -> torch.Tensor:
    """
    计算过程监督损失
    将评估分数转换为损失（分数越高，损失越小）
    Args:
        mask1_scores: [batch_size] mask1 的过程监督分数（1-100）
        mask2_scores: [batch_size] mask2 的过程监督分数（1-100）
        device: 设备
    Returns:
        loss: 标量损失值
    """
    # 将分数归一化到 [0, 1]
    mask1_weight = mask1_scores / 100.0
    mask2_weight = mask2_scores / 100.0
    
    # 过程监督损失：鼓励高质量推理路径
    # 使用负对数似然，分数越高损失越小
    eps = 1e-8
    ps_loss_mask1 = -torch.log(mask1_weight + eps)
    ps_loss_mask2 = -torch.log(mask2_weight + eps)
    
    return ps_loss_mask1.mean() + ps_loss_mask2.mean()

