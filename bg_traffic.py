import socket, pickle, time
import logging
import random
import numpy as np

from common import MyPacket, send_and_receive
from config import logger, HOST, PORT, INITIAL_RTT

DEBUG = False
MY_IP = 3


class BackgroundTraffic:
    def __init__(self):
        logger.debug("Initialising background traffic")
        self.client_socket = socket.socket()
        self.client_socket.settimeout(10)
        self.client_socket.connect((HOST, PORT))
        self.client_socket.send(
            pickle.dumps(MyPacket(0, 0, 0, true_source_ip=MY_IP, true_destination_ip=0))
        )
        logger.info("Background traffic generator connected")

    def send(self):
        self.num += 1
        # delay_list = [0.10, 0.50, 0.100, 0.500]
        delay_list = [0.010]
        time.sleep(random.choice(delay_list))
        bg_packet = MyPacket(
            size=4,
            source_ip=MY_IP,
            destination_ip=1,
            true_source_ip=MY_IP,
            true_destination_ip=1,
        )
        logger.debug(f"sent bg {self.num}")

        sent_time = time.time_ns()
        recv_data, finished = send_and_receive(
            self.client_socket, bg_packet, self.rtt, __name__
        )
        self.past_rtt_list.append((time.time_ns() - sent_time) / 1000000000)
        self.rtt = np.average(self.past_rtt_list)

        if finished:
            logger.debug("Closed via send().finished")
            self.client_socket.close()
        self._is_running = not finished

    def reset(self):
        logger.debug("Reset")
        self.num = 0
        self.past_rtt_list = []
        self.rtt = (
            INITIAL_RTT  # seconds. Large rtt time, not to owherwhelm the receiver
        )
        self._is_running = True

    def run(self):
        while self._is_running:
            self.send()

    def close(self):
        logger.debug("Closed via close()")
        self._is_running = False
        try:
            self.client_socket.close()
        except Exception:
            pass


if __name__ == "__main__":
    background_traffic = BackgroundTraffic()
    background_traffic.reset()
