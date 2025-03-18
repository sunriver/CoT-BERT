
import yaml
import os


def load_configs(default_file='', custom_file=''):
    # 深度合并字典（自定义配置优先）
    def merge_dict(d1, d2):
        for k, v in d2.items():
            if isinstance(v, dict) and k in d1:
                d1[k] = merge_dict(d1[k], v)
            else:
                d1[k] = v
        return d1

    config_default = {}
    if os.path.exists(default_file):
        with open(default_file, 'r') as f:
            config_default = yaml.safe_load(f)

    config_custom = {}
    if os.path.exists(custom_file):
        with open(custom_file, 'r') as f:
            config_custom = yaml.safe_load(f)

    config =  merge_dict(config_default, config_custom)

    args_list = []
    for key, value in config.items():
        # 处理布尔型参数（如--flag）
        if isinstance(value, bool):
            if value:
                args_list.append(f"--{key}")
        # 处理键值对参数（如--key value）
        else:
            args_list.extend([f"--{key}", str(value)])
    return args_list


def load_yaml_config(config_path):
    # 新增路径检查
    if not os.path.exists(config_path):
        raise ValueError(f"配置文件 {config_path} 不存在！")

    """加载YAML配置文件并转换为Hugging Face参数列表格式"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    args_list = []
    for key, value in config.items():
        # 处理布尔型参数（如--flag）
        if isinstance(value, bool):
            if value:
                args_list.append(f"--{key}")
        # 处理键值对参数（如--key value）
        else:
            args_list.extend([f"--{key}", str(value)])
    return args_list


if __name__ ==  "__main__":
    pass
    # load_yaml_config()
