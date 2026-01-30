"""
多变量分布检验：随机切片 + 单变量统计量。
对 (*, N, D) 做随机单位向量投影后，用单变量检验并聚合，兼容 mps/cuda/cpu。
"""
import torch
from .base import all_reduce


class SlicingUnivariateTest(torch.nn.Module):
    """
    Multivariate distribution test using random slicing and univariate test statistics.
    Projects (*, N, D) onto K random unit directions, runs a univariate test on each
    slice, then aggregates (mean/sum/None). Supports clip_value and distributed sync.
    """

    def __init__(
        self,
        univariate_test: torch.nn.Module,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        with torch.no_grad():
            global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()
            dev = dict(device=x.device)
            g = self._get_generator(x.device, seed)
            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)
        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats = stats.clone()
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        if self.reduction == "sum":
            return stats.sum()
        return stats
