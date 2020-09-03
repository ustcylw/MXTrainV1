import os, sys
import re, time
import logging
from logging.handlers import RotatingFileHandler


class Loggers:
    '''日志配置类'''

    log_colors = {
        'green': '\033[0;32;m',  # 绿色
        'blue': '\033[0;34;m',  # 蓝色
        'yellow': '\033[0;33;m',  # 黄色
        'navy': '\033[0;36;m',  # 藏青色
    }
    PROJECT_NAME = ''

    def __init__(self, logName='null', level='DEBUG'):
        self.logName = self.__get_log_path(logName)
        if not os.path.exists(os.path.dirname(self.logName)):
            os.makedirs(os.path.dirname(self.logName))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        if level == 'INFO': self.logger.setLevel(logging.INFO)
        self.formatter_console = logging.Formatter(
            f'{Loggers.log_colors["green"]}%(asctime)s | {Loggers.log_colors["blue"]}%(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d -> {Loggers.log_colors["navy"]}%(message)s')  # 日志输出格式-控制台
        self.formatter_local = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d -> %(message)s')  # 日志输出格式-文件
        # self.handle_logs()

    def __get_log_path(self, logName) -> str:
        '''
        设置log文件保存位置:
        日志存放路径:项目名称/logs/.
        '''
        path = os.getcwd()
        if re.search('markets-data', path):
            path = path.split('markets-data')[0] + 'markets-data/'
        else:
            path = path.split('markets')[0] + 'markets/'
        time_ymd = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        return f'{path}logs/{logName}_{time_ymd}.log'

    def get_logger(self):
        '''
        获取打印器
        解决重复打印问题:
            核心原因是每次调用longing,没有生成一个新的logger;
            方式一: 每次调用清空Handler
                    self.logger.handlers.clear()
            方式二: 向前判断
                    if not self.logger.handlers:pass
        '''

        # 解决重复打印2:判断是否存在Handler
        if not self.logger.handlers:
            # 创建一个FileHandler，用于写到日志文件
            fh = RotatingFileHandler(filename=self.logName, mode='a', maxBytes=1024 * 1024 * 5, backupCount=5,
                                     encoding='utf-8')  # 使用RotatingFileHandler类，滚动备份日志
            fh.setLevel(logging.INFO)
            fh.setFormatter(self.formatter_local)
            self.logger.addHandler(fh)

            # 创建一个StreamHandler,用于输出到控制台
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(self.formatter_console)
            self.logger.addHandler(ch)
        return self.logger


if __name__ == '__main__':

    logger = Loggers().get_logger()

    logger.info('hello')
    logger.debug('hello')
    logger.error('hello')
    logger.critical('hello')
