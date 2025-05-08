
class Strategy:
    max_seq_length = 32
    
    def __init__():
        pass
    
    @staticmethod
    def prepare_train_features(tokenizer, sentences):
        pass

    @staticmethod
    def prepare_eval_features(tokenizer, sentences):
        pass


    @staticmethod
    def get_pos_neg_pairs(denoised_mask_outputs):
        pass

    @staticmethod
    def get_sent_output(denoised_mask_outputs):
        pass

    @staticmethod
    def is_denoised_enabled():
        return  True