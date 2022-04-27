"""Config

Contains global configuration parameters.
"""
import logging
import sys
import socket


HOST = socket.gethostname()
PORT = 5434

PACKET_SIZE = 8196

INITIAL_RTT = 0.01  # seconds
MAX_RETRIES = 10  # times

timeout_multiplier = 2

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
