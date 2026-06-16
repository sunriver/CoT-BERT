"""
CoT prompt 解析与三视图 input 构建（对齐 BERTNew2/cot_bert_train.py）。
"""

from typing import List, Optional, Tuple


def parse_cot_template_to_strings(raw_template: str, mask_token: str) -> Tuple[str, str]:
    """
    将 CoT 模板字符串解析为 bs / es 文本片段。
    格式: *cls*_The_sentence_of_"*sent_0*"_means_*mask*_,_so_...
    """
    if not raw_template:
        return "", ""

    template = raw_template
    assert " " not in template, f"CoT template must not contain spaces: {raw_template!r}"

    template = (
        template.replace("*mask*", mask_token)
        .replace("*sep+*", "")
        .replace("*cls*", "")
        .replace("*sent_0*", " ")
    )
    parts = template.split(" ")
    bs = parts[0].replace("_", " ")
    es = parts[1].replace("_", " ") if len(parts) > 1 else ""
    return bs, es


def encode_cot_bs_es(tokenizer, bs_text: str, es_text: str) -> Tuple[List[int], List[int]]:
    """编码 bs/es，与 CoT-BERT prepare_features 拼接方式一致。"""
    bs = tokenizer.encode(bs_text, add_special_tokens=False)
    es = tokenizer.encode(es_text, add_special_tokens=False)
    if bs:
        bs = bs[:-1]
    if es:
        es = es[1:]
    return bs, es


def parse_all_cot_templates(model_args, tokenizer) -> None:
    """
    解析 model_args 上全部 CoT 模板，写入 mask_embedding_sentence_bs/es 等字段。
    """
    mask_token = tokenizer.mask_token

    if getattr(model_args, "mask_embedding_sentence_template", ""):
        bs, es = parse_cot_template_to_strings(
            model_args.mask_embedding_sentence_template, mask_token
        )
        model_args.mask_embedding_sentence_bs = bs
        model_args.mask_embedding_sentence_es = es

    diff = getattr(model_args, "mask_embedding_sentence_different_template", "") or ""
    if diff:
        bs2, es2 = parse_cot_template_to_strings(diff, mask_token)
        model_args.mask_embedding_sentence_bs2 = bs2
        model_args.mask_embedding_sentence_es2 = es2

    neg = getattr(model_args, "mask_embedding_sentence_negative_template", "") or ""
    if neg:
        bs3, es3 = parse_cot_template_to_strings(neg, mask_token)
        model_args.mask_embedding_sentence_bs3 = bs3
        model_args.mask_embedding_sentence_es3 = es3


def build_cot_input_ids(
    sentence: str,
    bs_ids: List[int],
    es_ids: List[int],
    tokenizer,
    max_seq_length: int,
) -> List[int]:
    sent_ids = tokenizer.encode(sentence, add_special_tokens=False)[:max_seq_length]
    return bs_ids + sent_ids + es_ids


def build_cot_views_for_sentences(
    sentences: List[str],
    model_args,
    tokenizer,
    max_seq_length: int,
) -> Tuple[dict, int]:
    """
    为每个句子构建 CoT 多视图 input_ids。
    返回 sent_features dict 与 num_sent。
    """
    bs, es = encode_cot_bs_es(
        tokenizer,
        model_args.mask_embedding_sentence_bs,
        model_args.mask_embedding_sentence_es,
    )

    has_diff = bool(getattr(model_args, "mask_embedding_sentence_different_template", ""))
    has_neg = bool(getattr(model_args, "mask_embedding_sentence_negative_template", ""))

    if has_diff:
        bs2, es2 = encode_cot_bs_es(
            tokenizer,
            model_args.mask_embedding_sentence_bs2,
            model_args.mask_embedding_sentence_es2,
        )
    else:
        bs2, es2 = bs, es

    if has_neg:
        bs3, es3 = encode_cot_bs_es(
            tokenizer,
            model_args.mask_embedding_sentence_bs3,
            model_args.mask_embedding_sentence_es3,
        )
    else:
        bs3, es3 = bs, es

    expanded = list(sentences)
    if has_neg:
        expanded = sentences + sentences + sentences
        num_views = 3
    elif has_diff:
        expanded = sentences + sentences
        num_views = 2
    else:
        num_views = 1

    total = len(sentences)
    sent_features = {"input_ids": [], "attention_mask": []}

    for i, sent in enumerate(expanded):
        if i < total:
            ids = build_cot_input_ids(sent, bs, es, tokenizer, max_seq_length)
        elif i < 2 * total:
            ids = build_cot_input_ids(sent, bs2, es2, tokenizer, max_seq_length)
        else:
            ids = build_cot_input_ids(sent, bs3, es3, tokenizer, max_seq_length)
        sent_features["input_ids"].append(ids)

    ml = max(len(x) for x in sent_features["input_ids"])
    pad_id = tokenizer.pad_token_id
    for i, ids in enumerate(sent_features["input_ids"]):
        sent_features["input_ids"][i] = ids + [pad_id] * (ml - len(ids))
        sent_features["attention_mask"].append([1] * len(ids) + [0] * (ml - len(ids)))

    return sent_features, num_views


def register_cot_templates_on_model(model, model_args, tokenizer, total_length: int = 80) -> None:
    """在 model 上注册 CoT 模板 token 与 denoising 所需字段。"""
    parse_all_cot_templates(model_args, tokenizer)

    model.pad_token_id = tokenizer.pad_token_id
    model.mask_token_id = tokenizer.mask_token_id
    model.mask_num = getattr(model_args, "mask_num", 2)
    model.total_length = total_length

    model.bs = tokenizer.encode(model_args.mask_embedding_sentence_bs, add_special_tokens=False)
    model.es = tokenizer.encode(model_args.mask_embedding_sentence_es, add_special_tokens=False)
    model.mask_embedding_template = tokenizer.build_inputs_with_special_tokens(model.bs + model.es)

    diff = getattr(model_args, "mask_embedding_sentence_different_template", "") or ""
    if diff:
        model.bs2 = tokenizer.encode(model_args.mask_embedding_sentence_bs2, add_special_tokens=False)
        model.es2 = tokenizer.encode(model_args.mask_embedding_sentence_es2, add_special_tokens=False)
        model.mask_embedding_template2 = tokenizer.build_inputs_with_special_tokens(
            model.bs2 + model.es2
        )

    neg = getattr(model_args, "mask_embedding_sentence_negative_template", "") or ""
    if neg:
        model.bs3 = tokenizer.encode(model_args.mask_embedding_sentence_bs3, add_special_tokens=False)
        model.es3 = tokenizer.encode(model_args.mask_embedding_sentence_es3, add_special_tokens=False)
        model.mask_embedding_template3 = tokenizer.build_inputs_with_special_tokens(
            model.bs3 + model.es3
        )


def build_single_cot_eval_input(
    sentence: str,
    model_args,
    tokenizer,
    max_seq_length: int = 128,
) -> List[int]:
    """评估/推理：单句 + 主 CoT 模板。"""
    if not hasattr(model_args, "mask_embedding_sentence_bs"):
        parse_all_cot_templates(model_args, tokenizer)
    bs, es = encode_cot_bs_es(
        tokenizer,
        model_args.mask_embedding_sentence_bs,
        model_args.mask_embedding_sentence_es,
    )
    return build_cot_input_ids(sentence, bs, es, tokenizer, max_seq_length)


def uses_cot_multi_view(model_args) -> bool:
    """是否启用 CoT 多视图（双 MASK + 至少 2 列模板）。"""
    if not getattr(model_args, "mask_embedding_sentence", False):
        return False
    mask_num = getattr(model_args, "mask_num", 1)
    has_diff = bool(getattr(model_args, "mask_embedding_sentence_different_template", ""))
    return mask_num >= 2 and has_diff
