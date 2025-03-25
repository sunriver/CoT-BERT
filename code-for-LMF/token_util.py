max_seq_length = 32

def prepare_train_features(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('The sentence of "')[:-1]
        es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[:-1]
        es_neg = tokenizer.encode('" does not mean [MASK] and [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos, bs + s + es_neg])
            sent_positions = ((len(bs), len(bs+s)), (len(bs), len(bs+s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features

def prepare_train_features1(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs1 = tokenizer.encode('The sentence of "')[:-1]
        bs2 = tokenizer.encode('The sentence : "')[:-1]
        es_pos = tokenizer.encode('" means [MASK], not means [MASK].')[:-1]
        es_neg = tokenizer.encode('" does not mean [MASK], but means [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs1 + s + es_pos, bs2 + s + es_neg])
            sent_positions = ((len(bs1), len(bs1+s)), (len(bs2), len(bs2+s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features


def prepare_eval_features(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('The sentence of "')[:-1]
        es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[:-1]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos])
            sent_positions = ((len(bs), len(bs+s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features

def prepare_eval_features1(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('The sentence of "')[:-1]
        es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[:-1]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos])
            sent_positions = ((len(bs), len(bs+s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features