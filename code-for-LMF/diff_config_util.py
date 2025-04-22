import token_util
import lmf_model
config1 = {
  "train_prepare_token": token_util.prepare_train_features,
   "eval_prepare_token": token_util.prepare_eval_features,
   "get_pos_neg_pairs": lmf_model.get_pos_neg_pairs,
   "get_sent_output": lmf_model.get_sent_output,
}

def get_config_funcs():
    return config1
