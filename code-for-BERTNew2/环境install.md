- conda create -n minicondaenv3_9_23 python=3.9.23 -y
- conda activate minicondaenv3_9_23
- pip install pip==25.3 -i https://pypi.tuna.tsinghua.edu.cn/simple //升级pip版本
- pip install -r requirement_new.txt
# 核心命令：指定 cu118 源，安装 torch/torchaudio/torchvision 对应版本
pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 torchvision==0.21.0+cu118 triton==3.2.0 \
-i https://download.pytorch.org/whl/cu118
