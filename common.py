"""Common

Contains common functions.
"""
import random
import pickle
from socket import socket, timeout
from math import floor

from config import timeout_multiplier, logger, MAX_RETRIES


"""
size - packet size in bits
source, destination - ip, max 32 bits
data - random data of packet
"""


class NetworkPacket:
    def __init__(
        self, size, source_ip, destination_ip, true_source_ip, true_destination_ip
    ):
        self.size = size
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.true_source_ip = true_source_ip
        self.true_destination_ip = true_destination_ip
        self.data = 0
        # self.fill_random()

    def fill_random(self):
        for _ in range(self.size):
            self.data += random.randint(0, 15)

    def str(self):
        str_info = f"--- Packet ---\n"
        str_info += f"size: {self.size}\n"
        str_info += f"src: {self.source_ip}\n"
        str_info += f"dst: {self.destination_ip}\n"
        str_info += f"true source: {self.true_source_ip}\n"
        str_info += f"true destination: {self.true_destination_ip}\n"
        str_info += "\n"
        return str_info


"""
1. Send packet
2. Receive response
3. Wait for rtt * timeout_multiplier time. If response is not received, 
   send the packet again.
4. If ran into an error, mark episode as finished and return empty.
"""


def send_and_receive(
    socket: socket,
    packet: NetworkPacket,
    rtt: float,
    source_file: str,
    max_attempts: int = MAX_RETRIES,
) -> tuple[str, int]:
    recv_response = False
    episode_finished = False
    num = 0
    rtt = rtt * timeout_multiplier
    recv_data = b""

    while not recv_response and num < max_attempts:
        num += 1
        logger.debug(f"  {source_file}: attempt {num}")

        # send the packet
        try:
            socket.send(pickle.dumps(packet))
        except (BrokenPipeError, IOError, InterruptedError) as e:
            logger.debug(f"  {source_file}: episode finished on send: {e}")
            episode_finished = True
            return "", episode_finished
        except Exception as e:
            logger.error(f"  {source_file}: Unexpected error when sending: {e}")
            episode_finished = True
            return "", episode_finished
        logger.debug(f"  {source_file}: packet sent")

        # wait and receive response
        try:
            if num < floor(MAX_RETRIES / 2):
                to_wait = rtt * num
            else:
                to_wait = (0.1 + rtt) * num
            logger.debug(f"timeout set to {to_wait}")
            socket.settimeout(to_wait)
            recv_data = socket.recv(4096)
            recv_response = True
            logger.debug(f"  {source_file}: response received")
        except (EOFError, BrokenPipeError, InterruptedError) as e:
            logger.debug(f"  {source_file}: episode finished on receive: {e}")
            episode_finished = True
            return recv_data, episode_finished
        except (TimeoutError, timeout):  # exception expected if no response
            pass
        except Exception as e:
            logger.error(f"  {source_file}: Unexpected error when receiving: {e}")

        if num == max_attempts:
            logger.debug(
                "Reached max connection attempts, marking episode as finished."
            )
            episode_finished = True
    return recv_data, episode_finished
