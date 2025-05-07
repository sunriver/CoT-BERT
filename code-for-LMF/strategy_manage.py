

from strategy1 import Strategy1
from strategy2 import Strategy2
from strategy3 import Strategy3


strategy_name = "cot10#10" #default

def set_strategy(name):
    strategy_name = name

def get_strategy():
    # if strategy_name == "cot10#10$11":
    #     return Strategy1
    # if strategy_name == "cot10#10$11":
    #         return Strategy1
    return Strategy3