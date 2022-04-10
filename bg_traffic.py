import socket, pickle, time
import logging

from common import MyPacket


HOST = socket.gethostname()
PORT = 5432
DEBUG = False


class BackgroundTraffic:
    def __init__(self):
        logging.debug("init")
