# from transformers.utils import logging
import logging

from datetime import datetime
from logging.handlers import RotatingFileHandler  # 支持日志文件滚动


class CustomLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.__init_config()
    
    def __init_config(self):
            # 获取当前时间并格式化
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 动态生成日志文件名
        log_file = f"run_{start_time}.log"
        log_file = "app1.log"

        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)  # 仅显示WARNING及以上级别
        self.addHandler(file_handler)
        self.addHandler(console_handler)

def initLogConfig():
    logging.setLoggerClass(CustomLogger)
    
def initLogConfig1():
    # print(dir(logging))
    # logging.config.fileConfig("log_config.ini")
    # logging.setLoggerClass(CustomLogger)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 动态生成日志文件名
    log_file = f"run_{start_time}.log"
    log_file = "app.log"

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # 仅显示WARNING及以上级别
    # 通过 basicConfig 同时添加两个 Handler
    logging.basicConfig(
        level=logging.INFO,  # 根日志级别（影响所有未单独设置级别的 Handler）
        handlers=[file_handler, console_handler]  # 指定多个 Handler
    )
   

def getMyLogger(name):
    # name = "stone"
    # logging.get_logger()
    logger = logging.getLogger(name)
    return logger
    # # Return logger if it has been inited, or init it before return
    # if hasattr(logger, "inited") and logger.inited:
    #     return logger

    # logger.inited = True
    # logger.setLevel(logging.INFO)

    # # 获取当前时间并格式化
    # start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # # 动态生成日志文件名
    # log_file = f"run_{logger.name}_{start_time}.log"

    # file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    # file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(file_formatter)

    # console_handler = logging.StreamHandler()
    # console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    # console_handler.setFormatter(console_formatter)
    # console_handler.setLevel(logging.INFO)  # 仅显示WARNING及以上级别

    # logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    # return logger
