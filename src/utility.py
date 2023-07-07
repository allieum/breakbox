import os
import sys
import logging

def get_logger(name):
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s %(threadName)s:%(funcName)s: %(message)s',
        # filename='break.log'
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)
