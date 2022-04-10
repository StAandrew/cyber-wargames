import socket, pickle, time
import logging
import random

from common import MyPacket, send_and_receive
from config import logger, HOST, PORT

DEBUG = False


class BackgroundTraffic:
    def __init__(self):
        logging.debug("init")
        self.client_socket = socket.socket()
        self.client_socket.settimeout(10)
        self.client_socket.connect((HOST, PORT))

    def send(self):
        self.num += 1
        # delay_list = [10, 50, 100, 500]
        delay_list = [10]
        time.sleep(random.choice(delay_list))
        bg_packet = MyPacket(size=4, src_ip=0, dst_ip=0, true_source=0)
        logger.debug(f"sent bg {self.num}")
        self.step_num += 1
        recv_data, self.finished = send_and_receive(self.client_socket, bg_packet, self.rtt)
    
    def reset(self):
        self.num = 0
        self.finished = False
        while not self.finished:
            self.send()

    def set_finished(self):
        self.finished = True