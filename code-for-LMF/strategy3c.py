

from strategy import Strategy

class Strategy3c(Strategy):

    @staticmethod
    def prepare_train_features(tokenizer, sentences):
            sent_features = {'input_ids': [], 'sent_positions': []}
            bs1 = tokenizer.encode("Let's think the sentence of '")[:-1]
            bs2 = tokenizer.encode("Let's think the sentence : '")[:-1]
            es_pos = tokenizer.encode("' step by step, it means [MASK].")[1:]
            # es_pos = tokenizer.encode('" means [MASK], so it can be summarized as [MASK].')[1:]
            es_neg = tokenizer.encode("' step by step, it does not mean [MASK].")[1:]
            # es_neg = tokenizer.encode("' means [MASK], but doesn't mean [MASK].")[1:]
            # es_neg = tokenizer.encode('" does not mean [MASK], and it also does not mean [MASK].')[1:]
            for i, sent in enumerate(sentences):
                if sent is None:
                    sent = " "
                s = tokenizer.encode(sent, add_special_tokens=False)[:Strategy.max_seq_length]
                sent_features['input_ids'].append([bs1 + s + es_pos, bs2 + s + es_pos, bs1 + s + es_neg])
                sent_positions = [(len(bs1), len(bs1 + s)), (len(bs2), len(bs2 + s)), (len(bs1), len(bs1 + s))]
                sent_features['sent_positions'].append(sent_positions)
            return sent_features

    @staticmethod
    def prepare_eval_features(tokenizer, sentences):
            sent_features = {'input_ids': [], 'sent_positions': []}
            bs = tokenizer.encode("Let's think the sentence of '")[:-1]
            # es_pos = tokenizer.encode('" not only implies [MASK] but also suggests [MASK].')[:-1]
            es_pos = tokenizer.encode("' step by step, it means [MASK].")[1:]
            # es_pos = tokenizer.encode("' means [MASK].")[1:]
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
    def get_pos_neg_pairs(denoised_mask_outputs):
        pos_mask1_vec = denoised_mask_outputs[:, 0, 0]
        pos_mask2_vec = denoised_mask_outputs[:, 1, 0]
        neg_mask1_vec = denoised_mask_outputs[:, 2, 0]

        pos_pairs = [(pos_mask1_vec, pos_mask2_vec)]
        neg_pairs = [
            (pos_mask1_vec, neg_mask1_vec),
            (pos_mask1_vec, neg_mask1_vec),
            # (neg_mask2_vec, neg_mask2_vec)
        ]
        
        return pos_pairs, neg_pairs

    @staticmethod
    def get_sent_output(denoised_mask_outputs):
        pos_mask_output_pooler = denoised_mask_outputs[:,0,:,:].squeeze(1) 
        # pos_mask_output_pooler, _ = pos_mask_output_pooler.max(dim = 1)
        # pos_mask_output_pooler= pos_mask_output_pooler.sum(dim = 1)
        # pos_mask_output_pooler= pos_mask_output_pooler.mean(dim = 1)
        return pos_mask_output_pooler


