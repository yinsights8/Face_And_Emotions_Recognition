import logging
import os
import sys
from datetime import datetime as dt
from logging.handlers import RotatingFileHandler

# This logger.py file is responsible for reporting errors when happens during runtime. 


LOG_FILE = f"{dt.now().strftime('%Y_%m_%d%I_%M_%S')}.log"
LOG_FORMAT = "[ %(asctime)s ] %(lineno)d  %(name)s - %(levelname)s - %(message)s"
log_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format=LOG_FORMAT,
#     level=logging.INFO
# )



logFormatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %I:%M:%S")

# send loggers to multiple streams
logger = logging.getLogger()

# add console handler to the root logger
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# add file handler to the root logger
fileHandler = RotatingFileHandler(filename=LOG_FILE_PATH, backupCount=100, maxBytes=1024)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)