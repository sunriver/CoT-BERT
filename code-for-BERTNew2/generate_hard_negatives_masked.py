"""
Generate high-quality hard negative samples by masking 20% of tokens and forcing BERT to predict a different token.
Strategy:
1. Input sentence 'sent'.
2. Randomly mask 20% of tokens (ensure at least one).
3. Feed the masked sentence into BertForMaskedLM.
4. For each [MASK] position:
   - If Top-1 prediction == original token, pick Top-2.
   - Else pick Top-1.
"""

import torch
from transformers import BertTokenizerFast, BertForMaskedLM
from tqdm import tqdm
import argparse
import os
import random
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except (LookupError, OSError):
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')

# Reuse device configuration from platform_utils if available
try:
    from platform_utils import setup_device_config
except ImportError:
    def setup_device_config():
        return {'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}

def mask_tokens_with_indices(input_ids, tokenizer, mask_prob=0.2, maskable_mask=None):
    """
    Randomly mask tokens with 20% probability.
    Returns:
        masked_ids: Tensor with [MASK] tokens.
        masked_indices: Boolean mask indicating which tokens were masked.
    """
    masked_ids = input_ids.clone()
    
    # Get special tokens mask
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in masked_ids.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    
    # Create probability matrix for masking
    probability_matrix = torch.full(masked_ids.shape, mask_prob)
    
    # Don't mask special tokens
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Only mask if it's in the maskable_mask (if provided)
    if maskable_mask is not None:
        # content_mask is [batch, seq_len]
        probability_matrix.masked_fill_(~maskable_mask, value=0.0)
    
    # Sample which tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Ensure at least one token is masked for each sentence if possible
    for i in range(masked_ids.shape[0]):
        if not masked_indices[i].any():
            # Find candidate indices (not special, and maskable if provided)
            candidates_mask = ~special_tokens_mask[i]
            if maskable_mask is not None:
                candidates_mask = candidates_mask & maskable_mask[i]
                
            candidates = candidates_mask.nonzero(as_tuple=False).squeeze(-1)
            if len(candidates) > 0:
                random_idx = candidates[random.randint(0, len(candidates) - 1)]
                masked_indices[i, random_idx] = True
            elif maskable_mask is not None:
                # If no content words found, fall back to any non-special token
                candidates = (~special_tokens_mask[i]).nonzero(as_tuple=False).squeeze(-1)
                if len(candidates) > 0:
                    random_idx = candidates[random.randint(0, len(candidates) - 1)]
                    masked_indices[i, random_idx] = True
                
    # Replace tokens with [MASK]
    masked_ids[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    return masked_ids, masked_indices

def generate_hard_negatives(input_file, output_file, model_path, batch_size, device):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(sentences)} sentences...")
    results = []
    
    content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'CD'}
    
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i : i + batch_size]
        
        # 1. NLTK Tokenization and POS Tagging for each sentence
        batch_words = []
        batch_maskable_word_indices = []
        for sent in batch_sentences:
            words = word_tokenize(sent)
            tags = nltk.pos_tag(words)
            maskable_indices = [idx for idx, (word, tag) in enumerate(tags) if tag in content_tags]
            batch_words.append(words)
            batch_maskable_word_indices.append(set(maskable_indices))
            
        # 2. BERT Tokenization using pre-split words
        encoded = tokenizer(
            batch_words, 
            is_split_into_words=True,
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # 3. Create maskable_mask at token level
        maskable_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(len(batch_sentences)):
            word_ids = encoded.word_ids(batch_index=b_idx)
            for t_idx, w_idx in enumerate(word_ids):
                if w_idx is not None and w_idx in batch_maskable_word_indices[b_idx]:
                    maskable_mask[b_idx, t_idx] = True
        
        # Mask tokens using the content-aware mask
        masked_input_ids, masked_indices = mask_tokens_with_indices(input_ids.cpu(), tokenizer, maskable_mask=maskable_mask.cpu())
        masked_input_ids = masked_input_ids.to(device)
        masked_indices = masked_indices.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits # [batch, seq_len, vocab_size]

        # Decode each sentence in batch
        for b_idx in range(len(batch_sentences)):
            sent_masked_indices = masked_indices[b_idx]
            curr_input_ids = input_ids[b_idx].clone()
            
            # Find mask positions for this sentence
            pos_indices = sent_masked_indices.nonzero(as_tuple=False).squeeze(-1)
            
            for pos in pos_indices:
                original_token_id = input_ids[b_idx, pos].item()
                
                # Get Top-6 predictions to ensure we have at least 5 candidates after excluding original
                top_6_values, top_6_indices = torch.topk(logits[b_idx, pos], k=6)
                candidates = [idx.item() for idx in top_6_indices if idx.item() != original_token_id]
                
                # Randomly pick one from Top-5 candidates (excluding original)
                predicted_token_id = random.choice(candidates[:5])
                
                curr_input_ids[pos] = predicted_token_id
            
            # Convert back to string
            negative_sentence = tokenizer.decode(curr_input_ids, skip_special_tokens=True)
            original_sentence = batch_sentences[b_idx]
            
            # Print samples with small probability for monitoring
            if random.random() < 0.001:
                print(f"\nOriginal: {original_sentence}")
                print(f"Negative: {negative_sentence}")
                print("-" * 50)
            
            results.append(f"{original_sentence}\t{negative_sentence}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + "\n")
            
    print(f"Done! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hard negatives using Masked Language Modeling and forced differentiation.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output TSV file.")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased", help="BERT model path.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    
    args = parser.parse_args()
    
    device_config = setup_device_config()
    device = torch.device(device_config['device'])
    
    generate_hard_negatives(args.input_file, args.output_file, args.model_path, args.batch_size, device)
