

from strategy1 import Strategy1
from strategy2 import Strategy2
from strategy3 import Strategy3
from strategy3c import Strategy3c
from strategy3c1 import Strategy3c1
from strategy4 import Strategy4


strategy_name = "cot10#10" #default

def set_strategy(name):
    strategy_name = name

def get_strategy():
    # if strategy_name == "cot10#10$11":
    #     return Strategy1
    # if strategy_name == "cot10#10$11":
    #         return Strategy1
    return Strategy3c1