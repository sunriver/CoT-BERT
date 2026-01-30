"""
单变量检验基类与分布式工具。
支持 mps / cuda / cpu，all_reduce 使用标准 torch.distributed API 以兼容不同 PyTorch 版本。
"""
import torch
from torch import distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def all_reduce(x, op="AVG"):
    """跨进程规约，支持 AVG / SUM / MAX。非分布式时直接返回 x。"""
    if not is_dist_avail_and_initialized():
        return x
    op = str(op).upper()
    if op == "AVG":
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / dist.get_world_size()
    elif op == "SUM":
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    elif op == "MAX":
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
    else:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        if op == "MEAN":
            x = x / dist.get_world_size()
    return x


class UnivariateTest(torch.nn.Module):
    """单变量检验基类，提供 world_size 与 dist_mean。"""

    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0, 1)

    def prepare_data(self, x):
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s

    def dist_mean(self, x):
        if is_dist_avail_and_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x = x / dist.get_world_size()
        return x

    @property
    def world_size(self):
        if is_dist_avail_and_initialized():
            return dist.get_world_size()
        return 1
