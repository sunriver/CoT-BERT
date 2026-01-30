# TexLeJEPA 核心统计检验模块 (SIGReg)
from .base import UnivariateTest, all_reduce
from .epps_pulley import EppsPulley
from .slicing import SlicingUnivariateTest

__all__ = ["UnivariateTest", "all_reduce", "EppsPulley", "SlicingUnivariateTest"]
