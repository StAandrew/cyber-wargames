import logging
import sys
import socket


HOST = socket.gethostname()
PORT = 5432

PACKET_SIZE = 8196

INITIAL_RTT = 1

models_dir = "models"
log_dir = "logs"
starts_with = "network"

logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# file_handler = logging.FileHandler('logfile.log')
# formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

timeout_multiplier = 1.1
