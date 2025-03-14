# 启动项目准备工作

## 配置https://hf-mirror.com/ 模型下载地址
参考链接https://hf-mirror.com/
huggingface-cli 是 Hugging Face 官方提供的命令行工具，自带完善的下载功能。
1. 安装依赖
pip install -U huggingface_hub
Copy
2. 设置环境变量
Linux
export HF_ENDPOINT=https://hf-mirror.com
建议将上面这一行写入 ~/.bashrc。

## 下载pretrain模型google-bert/bert-base-uncased
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir ./pretrain_models/bert-base-uncased
“./pretrain_models/bert-base-uncased”是自定义目录


## 配置/ect/hosts
# 3.167.112.25 huggingface.co
185.199.108.133 raw.githubusercontent.com
185.199.109.133 raw.githubusercontent.com
185.199.110.133 raw.githubusercontent.com
185.199.111.133 raw.githubusercontent.com

## 修改train.py
    args_list = [
        '--model_name_or_path', '/data/output/worksapce_gpu32/projects/pretrain_models/bert-base-uncased',
        '--train_file', '/data/output/worksapce_gpu32/projects/CoT-BERT/data/wiki1m_for_simcse.txt',
        '--output_dir', '../result/CoT-Bert', 
        # '--num_train_epochs', '10', 
        '--num_train_epochs', '1',
        # '--per_device_train_batch_size', '16', 
        '--per_device_train_batch_size', '4',
        '--learning_rate', '1e-5', 
        '--max_seq_length', '32', 
        '--evaluation_strategy', 'steps', 
        '--metric_for_best_model', 'stsb_spearman', 
        '--load_best_model_at_end', 
        '--eval_steps', '125', 
        '--overwrite_output_dir', 
        '--temp', '0.05', 
