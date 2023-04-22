import logging
import os
import sys
from datetime import datetime


def get_logdir_and_logger(filename='train.log'):
    now_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", now_time)
    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(logdir, filename),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, force=True, filemode='w')
    logger = logging.getLogger(__name__)
    # Avoid to add duplicate handlers
    if not len(logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logdir, logger