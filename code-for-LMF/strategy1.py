

from strategy import Strategy

class Strategy1(Strategy):

    @staticmethod
    def prepare_train_features(tokenizer, sentences):
        sent_features = {'input_ids': [], 'sent_positions': []}
        bs = tokenizer.encode("Let's think step by step, the sentence of '")[:-1]
        bs_neg = tokenizer.encode("Let's think step by step, the sentence : '")[:-1]
        es_pos = tokenizer.encode("' means [MASK], but doesn't mean [MASK].")[1:]
        # es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[1:]
        es_neg = tokenizer.encode("' means [MASK], but doesn't mean [MASK].")[1:]
        # es_neg = tokenizer.encode("' means [MASK], but doesn't mean [MASK].")[1:]
        # es_neg = tokenizer.encode('" does not mean [MASK], and it also does not mean [MASK].')[1:]
        for i, sent in enumerate(sentences):
            if sent is None:
                sent = " "
            s = tokenizer.encode(sent, add_special_tokens=False)[:Strategy.max_seq_length]
            sent_features['input_ids'].append([bs + s + es_pos, bs_neg + s + es_neg])
            sent_positions = ((len(bs), len(bs+s)), (len(bs_neg), len(bs_neg + s)))
            sent_features['sent_positions'].append(sent_positions)
        return sent_features

    @staticmethod
    def prepare_eval_features0(tokenizer, sentences):
            sent_features = {'input_ids': [], 'sent_positions': []}
            bs = tokenizer.encode("Let's think step by step, the sentence of '")[:-1]
            # es_pos = tokenizer.encode('" not only implies [MASK] but also suggests [MASK].')[:-1]
            # es_pos = tokenizer.encode("' means [MASK], and also means [MASK].")[1:]
            es_pos = tokenizer.encode("' means [MASK], but doesn't mean [MASK].")[1:]
            # es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[1:]
            for i, sent in enumerate(sentences):
                if sent is None:
                    sent = " "
                s = tokenizer.encode(sent, add_special_tokens=False)[:Strategy.max_seq_length]
                sent_features['input_ids'].append([bs + s + es_pos])
                sent_positions = ((len(bs), len(bs+s)),)
                sent_features['sent_positions'].append(sent_positions)
            return sent_features
    
    @staticmethod
    def prepare_eval_features(tokenizer, sentences):
        return Strategy1.prepare_train_features(tokenizer, sentences)


    @staticmethod
    def get_pos_neg_pairs(denoised_mask_outputs):
        pos_mask1_vec = denoised_mask_outputs[:, 0, 0]
        pos_mask2_vec = denoised_mask_outputs[:, 1, 0]
        neg_mask1_vec = denoised_mask_outputs[:, 0, 1]
        neg_mask2_vec = denoised_mask_outputs[:, 1, 1]

        pos_pairs = [(pos_mask1_vec, pos_mask2_vec)]
        neg_pairs = [
            (pos_mask1_vec, neg_mask1_vec),
            (pos_mask1_vec, neg_mask2_vec),
            (pos_mask2_vec, neg_mask1_vec),
            (pos_mask2_vec, neg_mask2_vec),
            (neg_mask1_vec, neg_mask2_vec),
            # (neg_mask2_vec, neg_mask2_vec)
        ]
        
        return pos_pairs, neg_pairs

    
    @staticmethod
    def get_sent_output(denoised_mask_outputs):
        pos_mask1_vec = denoised_mask_outputs[:, 0, 0]
        pos_mask2_vec = denoised_mask_outputs[:, 1, 0]
        pos_mask_vec = (pos_mask1_vec + pos_mask2_vec) / 2
        pos_mask_output_pooler = pos_mask_vec
        return pos_mask_output_pooler


    @staticmethod
    def get_sent_output0(denoised_mask_outputs):
        pos_mask_output_pooler = denoised_mask_outputs[:,0,:,:].squeeze(1) 
        # pos_mask_output_pooler, _ = pos_mask_output_pooler.max(dim = 1)
        # pos_mask_output_pooler= pos_mask_output_pooler.sum(dim = 1)
        pos_mask_output_pooler= pos_mask_output_pooler.mean(dim = 1)
        # pos_mask_output_pooler = pos_mask_output_pooler[:, 0, :]
        return pos_mask_output_pooler
    
    @staticmethod
    def get_sent_output1(denoised_mask_outputs):
        pos_mask_output_pooler = denoised_mask_outputs[:,0,:,:].squeeze(1) 
        # pos_mask_output_pooler, _ = pos_mask_output_pooler.max(dim = 1)
        # pos_mask_output_pooler= pos_mask_output_pooler.sum(dim = 1)
        # pos_mask_output_pooler= pos_mask_output_pooler.mean(dim = 1)
        pos_mask_output_pooler = pos_mask_output_pooler[:, 0, :]
        return pos_mask_output_pooler


