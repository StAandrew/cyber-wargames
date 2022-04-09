import random


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
