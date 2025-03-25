max_seq_length = 32

def prepare_train_features(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('The sentence of "')[:-1]
        es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[1:]
        es_neg = tokenizer.encode('" does not mean [MASK] or [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos, bs + s + es_neg])
            sent_positions = ((len(bs), len(bs+s)), (len(bs), len(bs+s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features


def prepare_eval_features(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode('The sentence of "')[:-1]
        # es_pos = tokenizer.encode('" not only implies [MASK] but also suggests [MASK].')[:-1]
        # es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[:-1]
        es_pos = tokenizer.encode('" means [MASK], and also means [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos])
            sent_positions = ((len(bs), len(bs+s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features

