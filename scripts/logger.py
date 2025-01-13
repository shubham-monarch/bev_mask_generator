import logging
import sys
import coloredlogs

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    '''Get a logger with colored output and line numbers
    
    Args:
        name: name of the logger
        level: logging level (default: logging.INFO)
        
    Returns:
        logger: configured logger instance with colored output and line numbers
    '''
    
    logging.basicConfig(level=level)
    logger = logging.getLogger(name)
    logger.propagate = False

    # define format with line numbers and module name
    log_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    date_format = "%Y/%m/%d %H:%M:%S"

    # colored console handler
    console_handler = logging.StreamHandler(sys.stdout)
    colored_formatter = logging.Formatter(
        fmt=f"\x1b[32m%(asctime)s\x1b[0m [%(levelname)s] %(name)s:\x1b[36m%(lineno)d\x1b[0m - %(message)s",
        datefmt=date_format
    )
    console_handler.setFormatter(colored_formatter)
    logger.handlers = [console_handler]
    
    # install colored logs
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt=log_format,
        datefmt=date_format,
        force=True
    )

    return logger    
