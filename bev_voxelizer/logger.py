import logging
import sys
import coloredlogs

def get_logger(name, level=logging.INFO):
    '''Get a logger with colored output'''
    
    logging.basicConfig(level=level)
    logger = logging.getLogger(name)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    consoleHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="\x1b[32m%(asctime)s\x1b[0m %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    consoleHandler.setFormatter(formatter)
    logger.handlers = [consoleHandler]
    coloredlogs.install(level=level, logger=logger, force=True)

    return logger    
