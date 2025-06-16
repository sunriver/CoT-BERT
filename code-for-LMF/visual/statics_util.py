
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理NumPy数值类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        # 处理NumPy数组
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # 处理其他类型（如datetime）
        # elif isinstance(obj, datetime):
        #     return obj.isoformat()
        return super().default(obj)

def save_results(results, json_file_name):
    with open(json_file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)