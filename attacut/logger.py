import logging
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=os.getenv("ATTACUT_LOG_LEVEL", "warning").upper()
)


def get_logger(name):
    return logging.getLogger(name)
