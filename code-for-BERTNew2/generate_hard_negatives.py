"""
Generate hard negative samples using a masked language model (BERT) and a template-based approach.

This script implements the strategy: 
'The sentence "{sent_0}" does not mean the sentence "{sent_0_mask}"'
where {sent_0_mask} is a randomly masked version of {sent_0}. 
The model fills in the masks to create a sentence that is lexically similar 
but semantically different (hard negative).

Usage in code-for-BERTNew2 environment:
    python generate_hard_negatives.py \
        --input_file data/raw_sentences.txt \
        --output_file data/hard_negatives.tsv \
        --model_path bert-base-uncased \
        --batch_size 32

Dependencies: torch, transformers, tqdm, platform_utils
"""

import torch
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import argparse
import os
import random

# Reuse device configuration from platform_utils
from platform_utils import detect_platform, setup_device_config

def mask_tokens(input_ids, tokenizer, mask_prob=0.2):
    """
    Randomly mask tokens with 20% probability.
    Ensure at least one token is masked (if input_ids is not empty and contains non-special tokens).
    Skip special tokens ([CLS], [SEP], [PAD]).
    """
    # Create a copy to avoid modifying the original
    masked_ids = input_ids.clone()
    
    # Get special tokens indices
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in masked_ids.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    
    # Create probability matrix for masking
    probability_matrix = torch.full(masked_ids.shape, mask_prob)
    # Don't mask special tokens
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Sample which tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Ensure at least one token is masked for each sentence if it's not all special tokens
    for i in range(masked_ids.shape[0]):
        # Check if any token is masked in this sentence
        if not masked_indices[i].any():
            # Find candidate indices (not special tokens)
            candidates = (~special_tokens_mask[i]).nonzero(as_tuple=False).squeeze(-1)
            if len(candidates) > 0:
                # Randomly pick one candidate to mask
                random_idx = candidates[random.randint(0, len(candidates) - 1)]
                masked_indices[i, random_idx] = True
                
    # Replace tokens with [MASK] where masked_indices is True
    masked_ids[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    return masked_ids, masked_indices

def generate_hard_negatives(input_file, output_file, model_path, batch_size, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
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
    
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i : i + batch_size]
        
        # We process each sentence in the batch. 
        # Note: While we iterate over batch_sentences, the model inference 
        # is currently done per-sentence within this loop to handle the template 
        # and mask extraction correctly for each individual sentence.
        batch_outputs = []
        for sent in batch_sentences:
            # 1. Mask tokens in sent_0
            sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
            sent_input_ids = torch.tensor([sent_tokens])
            
            # Mask sent_0 with 20% probability
            sent_masked_ids, mask_indices = mask_tokens(sent_input_ids, tokenizer, mask_prob=0.3)
            
            # 2. Build the template for this single sentence
            # [CLS] The sentence " {sent_0} " does not mean the sentence " {sent_0_mask} " [SEP]
            # Since we want to use the model to fill in the masks in sent_0_mask, 
            # we need to construct a sequence where sent_0_mask is actually masked.
            
            sent_0_tokens = tokenizer.convert_ids_to_tokens(sent_tokens)
            sent_0_mask_tokens = tokenizer.convert_ids_to_tokens(sent_masked_ids[0].tolist())
            
            masked = tokenizer.convert_tokens_to_string(sent_0_mask_tokens)
            template_text = f"The sentence \"{sent}\" contradicts the sentence \"{masked}\""
            
            should_print = random.random() < 0.01
            if should_print:
                print(f"Template: {template_text}")

            # Tokenize the whole template
            template_encoded = tokenizer(template_text, return_tensors="pt").to(device)
            
            # Find the positions of [MASK] in the template_encoded['input_ids']
            mask_token_id = tokenizer.mask_token_id
            mask_positions = (template_encoded['input_ids'] == mask_token_id).nonzero(as_tuple=False)[:, 1]
            
            if len(mask_positions) == 0:
                # Fallback if no mask was generated (shouldn't happen with our mask_tokens logic)
                batch_outputs.append(sent)
                continue
                
            with torch.no_grad():
                outputs = model(**template_encoded)
                predictions = outputs.logits
            
            # Extract predicted tokens for the masked positions
            predicted_ids = template_encoded['input_ids'].clone()
            for pos in mask_positions:
                predicted_token_id = torch.argmax(predictions[0, pos]).item()
                predicted_ids[0, pos] = predicted_token_id
            
            # We want to extract ONLY the predicted version of sent_0_mask
            # The template is: [CLS] The sentence " {sent_0} " does not mean the sentence " {sent_0_mask} " [SEP]
            # We can find the second occurrence by looking at the structure.
            # However, it's easier to just decode the whole thing and extract the part after "does not mean the sentence "
            
            decoded_full = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            # Find the part after "contradicts the sentence"
            marker = "contradicts the sentence"
            if marker in decoded_full:
                negative_sentence = decoded_full.split(marker)[-1].strip()
                # Clean up quotes if present
                negative_sentence = negative_sentence.strip('"').strip()
                if should_print:
                    print(f"Original: {sent}")
                    print(f"Negative: {negative_sentence}")
                    print("-" * 50)
                batch_outputs.append(negative_sentence)
            else:
                if should_print:
                    print(f"Fallback used for: {sent}")
                batch_outputs.append(sent) # Fallback

        for orig, neg in zip(batch_sentences, batch_outputs):
            results.append(f"{orig}\t{neg}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + "\n")
            
    print(f"Done! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hard negatives using BERT and 'does not mean' template.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file (one sentence per line).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output TSV file.")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased", help="BERT model path or name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    
    args = parser.parse_args()
    
    # Setup device
    device_config = setup_device_config()
    device = torch.device(device_config['device'])
    
    generate_hard_negatives(args.input_file, args.output_file, args.model_path, args.batch_size, device)
