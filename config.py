import logging
import sys
import socket


HOST = socket.gethostname()
PORT = 5432

models_dir = "models"
log_dir = "logs"
starts_with = "network"

logger = logging.Logger()
logger.basicConfig(stream=sys.stderr, level=logging.DEBUG)
