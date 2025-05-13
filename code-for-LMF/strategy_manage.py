

from strategy import Strategy
from strategy1 import Strategy1
from strategy2 import Strategy2
from strategy3 import Strategy3
from strategy3c import Strategy3c
from strategy3c1 import Strategy3c1
from strategy3c2 import Strategy3c2
from strategy3c2c1 import Strategy3c2c1
from strategy3c3 import Strategy3c3

from importlib import import_module


strategy_name = "strategy3" #default
strategry_module = "strategy3"
strategry_module_class = "strategy3"

def set_strategy(name):
    name = "strategy3"
    # strategy_name = name
    # strategry_module = import_module(name)
    # class_name= name.title() #首字母大写
    # strategry_module_class =  getattr(strategry_module, class_name)

def get_strategy():
    # if strategy_name == "cot10#10$11":
    #     return Strategy1
    # if strategy_name == "cot10#10$11":
    #         return Strategy1
    return Strategy3c2c1