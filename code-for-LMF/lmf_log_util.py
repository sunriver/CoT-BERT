import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler  # 支持日志文件滚动


def getMyLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 获取当前时间并格式化
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 动态生成日志文件名
    log_file = f"run_{logger.name}_{start_time}.log"

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)


    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING)  # 仅显示WARNING及以上级别

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger