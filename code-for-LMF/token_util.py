max_seq_length = 64
import random

mask_token_ids = {
    "mask": -1,
    "mask1": -1,
    "mask2": -1
}

def init_my_special_tokens(model, tokenizer):
    # 添加新 token
    new_tokens = ["[MASK1]", "[MASK2]"]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    print("new special tokens added:", tokenizer.special_tokens_map)
    # 调整模型嵌入层大小
    model.resize_token_embeddings(len(tokenizer))

    mask_token_ids['mask'] = tokenizer.convert_tokens_to_ids("[MASK]") 
    mask_token_ids['mask1'] = tokenizer.convert_tokens_to_ids("[MASK1]") 
    mask_token_ids['mask2'] = tokenizer.convert_tokens_to_ids("[MASK2]") 

    return tokenizer

def get_mask_token_id(): 
    return  mask_token_ids['mask'],  mask_token_ids['mask1'],  mask_token_ids['mask2']

    

def prepare_train_features0(tokenizer, sentences):
        sep_token_id = tokenizer.sep_token_id
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('The sentence : "')[:-1]
        # bs1 = tokenizer.encode('The sentence : "')[:-1]
        es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[1:]
        # es_pos = tokenizer.encode('" means [MASK] and [MASK].')[1:]
        # es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[1:]
        # es_neg = tokenizer.encode('" does not mean [MASK] or [MASK].')[1:]
        es_neg = tokenizer.encode('" means [MASK], but does not mean [MASK].')[1:]
        # es_neg = tokenizer.encode('" does not mean [MASK], and it also does not mean [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            # s = [sep_token_id, *s, sep_token_id]
            
            sent_features['input_ids'].append([bs + s + es_pos, bs + s + es_neg])
            sent_positions = ((len(bs), len(bs+s)), (len(bs), len(bs + s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features


def prepare_train_features2(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('The sentence : "')[:-1]
        es_pos = tokenizer.encode('" means [MASK1], but does not means [MASK2].')[1:]
        # es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[1:]
        es_neg = tokenizer.encode('" does not mean [MASK2], but means [MASK1].')[1:]
        # es_neg = tokenizer.encode('" does not mean [MASK], and it also does not mean [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos, bs + s + es_neg])
            sent_positions = ((len(bs), len(bs+s)), (len(bs), len(bs + s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features

def prepare_train_features(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('Let\'s think step by step, the sentence of "')[:-1]
        es_pos = tokenizer.encode('" means [MASK], but does not means [MASK].')[1:]
        # es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[1:]
        es_neg = tokenizer.encode('" does not mean [MASK], but means [MASK].')[1:]
        # es_neg = tokenizer.encode('" does not mean [MASK], and it also does not mean [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos, bs + s + es_neg])
            sent_positions = ((len(bs), len(bs+s)), (len(bs), len(bs + s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features

def prepare_train_features5(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('What does the sentence of "')[:-1]
        es_pos = tokenizer.encode('" mean, it means [MASK], but does not mean [MASK].')[1:]
        # es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[1:]
        es_neg = tokenizer.encode('" it does not mean [MASK], but means [MASK].')[1:]
        # es_neg = tokenizer.encode('" does not mean [MASK], and it also does not mean [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos, bs + s + es_neg])
            sent_positions = ((len(bs), len(bs+s)), (len(bs), len(bs + s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features

def prepare_eval_features0(tokenizer, sentences):
        sep_token_id = tokenizer.sep_token_id
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('The sentence : "')[:-1]
        # es_pos = tokenizer.encode('" not only implies [MASK] but also suggests [MASK].')[:-1]
        # es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[:-1]
        # es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[1:]
        es_pos = tokenizer.encode('" means [MASK].')[1:]
        # es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            # s = [sep_token_id, *s, sep_token_id]
            sent_features['input_ids'].append([bs + s + es_pos])
            sent_positions = ((len(bs), len(bs+s)),)
            sent_features['sent_positions'].append(sent_positions)
        return sent_features

def prepare_eval_features(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('Let\'s think step by step, the sentence of "')[:-1]
        # es_pos = tokenizer.encode('" not only implies [MASK] but also suggests [MASK].')[:-1]
        # es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[:-1]
        es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[1:]
        # es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos])
            sent_positions = ((len(bs), len(bs+s)),)
            sent_features['sent_positions'].append(sent_positions)
        return sent_features

# def prepare_eval_features(tokenizer, sentences):
#     return prepare_train_features(tokenizer, sentences)
