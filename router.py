import logging
import socket
import threading
import pickle

from config import logger, HOST, PORT, PACKET_SIZE
from common import MyPacket


"""
Behaviour: if a need client with same ip connects, it replaces 
  old client with that ip address.
"""


class Router:
    def __init__(self):
        logger.info("Router started")
        bind_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bind_socket.bind((HOST, PORT))
        bind_socket.settimeout(10)
        bind_socket.setblocking(1)

        self.max_conn_num = 100
        self.conn_num = 0
        self.connections = {}
        while True:
            bind_socket.listen(self.max_conn_num)
            conn, addr = bind_socket.accept()
            logger.info(f"Connection initialised from: {str(addr)}")

            try:
                recv_data = conn.recv(PACKET_SIZE)
                packet = pickle.loads(recv_data)
                source_ip = int(packet.true_source_ip)
            except EOFError as e:
                logger.warning(f"Warning: {e}")
            except Exception as e:
                logger.warning(f"Unexpected warning: {e}")
            try:
                self.connections[source_ip]
            except KeyError:  # This source_ip was not assigned yet
                self.conn_num += 1
            self.connections[source_ip] = (conn, addr)
            logger.info(f"Assigned ip: {source_ip}")

            threading.Thread(target=self.connection_listen, args=(source_ip,)).start()

    def connection_listen(self, source_ip):
        (conn, addr) = self.connections[source_ip]
        logger.info(f"Running in background, listening for ip {source_ip}")
        logger.debug(f"{conn}, {addr}")
        while True:
            try:
                data_raw = conn.recv(PACKET_SIZE)
                if data_raw != b"":
                    data = pickle.loads(data_raw)
                    if isinstance(data, MyPacket):
                        destination_ip = int(data.true_destination_ip)
                    else:
                        data = dict(data)
                        # logger.debug(f"Dict: {data}")
                        destination_ip = int(data.get("true_destination_ip"))
                    try:
                        self.connections[destination_ip]
                        try:
                            dest_conn = self.connections[destination_ip][0]
                            dest_conn.send(data_raw)
                        except TypeError as e:
                            logger.error(
                                f"Packet from ip {source_ip} was not forwarded, host not found with ip {destination_ip}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error when sending packet received from ip {source_ip} to ip {destination_ip}: {e}"
                            )
                    except KeyError:  # This source_ip was not assigned yet
                        logger.error(
                            f"Packet from ip {source_ip} was not forwarded, host not found with ip {destination_ip}"
                        )
            except EOFError as e:
                logger.warning(f"Warning - pickle ran out of data: {e}")
                data_raw = b""
            except ConnectionResetError as e:
                logger.debug(f"Connection reset: {e}")
                self.connections[source_ip] = None
                exit(0)
            except Exception as e:
                logger.critical(
                    f"Critical error when receiving packet on ip {source_ip}: {e}"
                )
                self.connections[source_ip] = None
                exit(1)


if __name__ == "__main__":
    router = Router()
