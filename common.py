import random
import pickle

from config import timeout_multiplier

"""
size - packet size in bits
source, destination - ip, max 32 bits
data - random data of packet
"""


class MyPacket:
    def __init__(self, size, src_ip, dst_ip, true_source):
        self.size = size
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.data = 0
        self.fill_random()
        self.true_source = true_source

    def fill_random(self):
        for _ in range(self.size):
            self.data += random.randint(0, 15)

    def str(self):
        str_info = f"--- Packet ---\n"
        str_info += f"size: {self.size}\n"
        str_info += f"src: {self.src_ip}\n"
        str_info += f"dst: {self.dst_ip}\n"
        str_info += f"true source: {self.true_source}\n"
        str_info += "\n"
        return str_info


"""
1. Send packet
2. Receive response
3. Wait for rtt * timeout_multiplier time. If response is not received, 
   send the packet again.
4. If ran into an error, mark episode as finished and return empty.
"""
def send_and_receive(logger, socket, packet, rtt, max_attempts = 10):
    recv_response = False
    episode_finished = False
    num = 0
    rtt = rtt * timeout_multiplier

    while not recv_response and num < max_attempts:
        num += 1
        logger.debug(f"  send_rcv_func: attempt {num}")

        # send the packet
        try:
            socket.send(pickle.dumps(packet))
        except BrokenPipeError as e:
            logger.debug(f"  send_rcv_func: episode finished on send: {e}")
            episode_finished = True
            return "", episode_finished
        except Exception as e:
            logger.error("  send_rcv_func: Unexpected error when sending: {e}")
            episode_finished = True
            return "", episode_finished
        logger.debug("  send_rcv_func: packet sent")

        # wait and receive response
        socket.settimeout(rtt)
        try:
            recv_data = socket.recv(4096)
            recv_response = True
            logger.debug("  send_rcv_func: response received")
        except (EOFError, BrokenPipeError) as e:
            logger.debug("  send_rcv_func: episode finished on receive: {e}")
            episode_finished = True
            return recv_data, episode_finished
        except TimeoutError:   # exception expected if no response
            pass
        except Exception as e:
            logger.error(f"  send_rcv_func: Unexpected error when receiving: {e}")
    return recv_data, episode_finished
