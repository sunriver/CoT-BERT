python generate_hard_negatives_masked.py \
    --input_file ../data/wiki1m_for_simcse.txt \
    --output_file ../data/hard_negatives_masked.tsv \
    --batch_size 256 \
    --model_path /data/output/workspace_gpu3/pretrain_models/bert-base-uncased