import functools
import sys
from loguru import logger as loguru_logger

# 定义自定义的日志等级
loguru_logger.level("IMPORTANT", no=100, color="<blue><bold><italic>")

def logger_wraps(level):
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            # 提取第二个参数作为日志消息
            message = args[0] if len(args) > 0 else ""
            self._log(level, message)
            return func(self, *args, **kwargs)  # 确保原始方法仍然被调用
        return wrapped
    return wrapper

class Logger:
    def __init__(self):
        self.logger = loguru_logger

    def _log(self, level, message):
        # 通用的日志记录方法
        logger_ = self.logger.opt(depth=2)
        logger_.log(level, message)

    @logger_wraps(level="TRACE")
    def trace(self, message):
        pass

    @logger_wraps(level="DEBUG")
    def debug(self, message):
        pass

    @logger_wraps(level="INFO")
    def info(self, message):
        pass

    @logger_wraps(level="SUCCESS")
    def success(self, message):
        pass

    @logger_wraps(level="WARNING")
    def warning(self, message):
        pass

    @logger_wraps(level="ERROR")
    def error(self, message):
        pass

    @logger_wraps(level="CRITICAL")
    def critical(self, message, exit_code=1):
        sys.exit(exit_code)

    @logger_wraps(level="IMPORTANT")
    def important(self, message):
        pass

logger = Logger()

def get_logger():
    return logger

"""使用示例"""
if __name__ == '__main__':
    logger.info("This is an info message.")
    logger.success("SUCCESS1111111")
    logger.warning("WARNING")
    logger.success("SUCCESS")
    logger.error("ERROR")
    logger.important("This is an important message that must always be output.")
    logger.error("ERROR")
    logger.critical("This is a critical error. Exiting the program.")
    logger.important("This is an important message that must always be output.")
