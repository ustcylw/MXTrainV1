import os
import sys
import time
import logging
from logging.handlers import TimedRotatingFileHandler

# 日志级别
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CURRENT_PATH, os.pardir)
LOG_PATH = os.path.join(ROOT_PATH, 'logs')

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
COLORS_LEVELS_MAP = {
    'WARNING': MAGENTA,
    'INFO': GREEN,  # WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        # message = record.getMessage()
        if self.use_color and levelname in COLORS_LEVELS_MAP:
            levelname_color = COLOR_SEQ % (30 + COLORS_LEVELS_MAP[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
            # message_color = COLOR_SEQ % (30 + COLORS_LEVELS_MAP[levelname]) + message + RESET_SEQ
            # record.message = message_color
        return logging.Formatter.format(self, record)


class LogHandler(logging.Logger):
    """
    LogHandler
    """
    
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            # 一开始居然用了 cls()来实例化 导致无限次调用
            # cls._instance = cls(*args, **kwargs)
            # cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, name='%s' % time.strftime('%Y-%m-%d-%H-%M-%S'), level=DEBUG, stream=True, file=True, save_dir=ROOT_PATH, fixed_width=16):
        super(LogHandler, self).__init__(name=name, level=level)
        self.name = name
        self.level = level
        self.save_dir = save_dir if len(save_dir) > 0 else ROOT_PATH
        self.log_path = os.path.join(self.save_dir, 'logs')
        self.fixed_width = fixed_width
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        # if len(os.listdir(self.log_path)) > 0:
            # os.system('rm {}/*.log'.format(self.log_path))
        logging.Logger.__init__(self, self.name, level=level)
        if stream:
            self.__setStreamHandler__()
        if file:
            self.__setFileHandler__()

    def formatter_message(self, message, use_color=True):
        if use_color:
            message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
        else:
            message = message.replace("$RESET", "").replace("$BOLD", "")
        return message

    def __get_formatter(self):
        pass

    import colorlog
    def __setFileHandler__(self, level=None):
        """
        set file handler
        :param level:
        :return:
        """
        file_name = os.path.join(self.log_path, '{name}.log'.format(name=self.name))
            
        # 设置日志回滚, 保存在log目录, 一天保存一个文件, 保留15天
        file_handler = TimedRotatingFileHandler(filename=file_name, when='D', interval=1, backupCount=15)
        formatter = logging.Formatter('%(asctime)s %(filename)s:%(funcName)s:%(lineno)d [%(levelname)s] %(message)s')
        # FORMAT = f"[%(asctime)s] [$BOLD%(name)-{self.fixed_width}s$RESET] [%(levelname)s]  %(message)s %(filename)s:%(funcName)s:%(lineno)d"
        # # FORMAT = f"[%(asctime)s] [$BOLD%(name)-{self.fixed_width}s$RESET] [%(levelname)s]  %(message)s $BOLD[%(filename)s:%(funcName)s:%(lineno)d]$RESET"
        # COLOR_FORMAT = self.formatter_message(FORMAT, True)
        # formatter = ColoredFormatter(COLOR_FORMAT)
        file_handler.setFormatter(formatter)

        file_handler.suffix = '%Y%m%d.log'
        if not level:
            file_handler.setLevel(self.level)
        else:
            file_handler.setLevel(level)

        self.file_handler = file_handler
        self.addHandler(file_handler)

    def __setStreamHandler__(self, level=None):
        """
        set stream handler
        :param level:
        :return:
        """
        stream_handler = logging.StreamHandler()
        FORMAT = f"[%(asctime)s] [$BOLD%(name)-{self.fixed_width}s$RESET] [%(levelname)s]  %(message)s $BOLD[%(filename)s:%(funcName)s:%(lineno)d]$RESET"
        COLOR_FORMAT = self.formatter_message(FORMAT, True)
        formatter = ColoredFormatter(COLOR_FORMAT)

        stream_handler.setFormatter(formatter)
        if not level:
            stream_handler.setLevel(self.level)
        else:
            stream_handler.setLevel(level)
        self.addHandler(stream_handler)


    def resetName(self, name):
        """
        reset name
        :param name:
        :return:
        """
        self.name = name
        self.removeHandler(self.file_handler)
        self.__setFileHandler__()


if __name__ == '__main__':

    logger1 = LogHandler('logger')
    logger2 = LogHandler('logger')
    # print(logger1, id(logger1), logger2, id(logger2))
    logger1.info(f'haha  {id(logger1)}')
    logger2.info(f'hehe  {id(logger1)}')

    logger1.debug(f'haha  {id(logger1)}')
    logger2.warning(f'hehe  {id(logger1)}')
    logger1.critical(f'hehe  {id(logger1)}')
    logger2.error(f'hehe  {id(logger1)}')
