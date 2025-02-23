import logging
import sys

# Define ANSI escape codes for colors
RESET = "\033[0m"
COLORS = {
    'DEBUG': "\033[1m\033[94m",  # Blue
    'INFO': "\033[1m\033[38;5;208m",   # Orange
    'WARNING': "\033[1m\033[93m",# Yellow
    'ERROR': "\033[1m\033[91m",  # Red
    'CRITICAL': "\033[1m\033[95m",# Magenta
    'GREEN': "\033[1m\033[92m",    # Green
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, RESET)
        record.name = f"{log_color}{record.name}{RESET}"
        record.levelname = f"{log_color}{record.levelname}{RESET}"
        record.msg = f"{log_color}{record.msg}{RESET}"
        return super().format(record)

def get_logger(name=None):
    if name is None:
        # Get the caller's module name
        name = sys._getframe(1).f_globals['__name__']

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    # file_handler = logging.FileHandler(f'{name}.log')

    # Set log level for handlers
    console_handler.setLevel(logging.DEBUG)
    # file_handler.setLevel(logging.WARNING)

    # Create formatters and add them to the handlers
    console_format = ColoredFormatter('\n%(name)s - %(levelname)s\n%(message)s\n')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(console_format)
    # file_handler.setFormatter(file_format)

    # Add handlers to the logger
    if not logger.handlers:  # To avoid adding handlers multiple times
        logger.addHandler(console_handler)
        # logger.addHandler(file_handler)

    # To prevent logging from propagating to the root logger
    logger.propagate = False

    return logger