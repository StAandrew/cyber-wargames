"""Router

This script allows multiple clients to communicate with each other.
It can be run in background and supports silmultanious connections from 
multiple IP addresses. 

IP address is assigned to a newly connected client based on the first packet
received from the client. If a new client connects claiming the same IP address
it replaces the old client that had the same IP address. 

If router receives a packet that needs to be forwarded to an unassigned IP, it
raises a warning that a client with this IP address cannot be found and drops 
the packet. Clients disconnecting or dropping the connection are handled
automatically and do not raise an exception.

The scrips is based on threading and a maximum number of connections can be
specified by MAX_CONNECTIONS variable.
"""
import logging
import socket
import threading
import pickle

from config import logger, HOST, PORT, PACKET_SIZE
from common import NetworkPacket


MAX_CONNECTIONS = 100


class Router:
    def __init__(self):
        logger.info("Router started")
        bind_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bind_socket.bind((HOST, PORT))
        bind_socket.settimeout(10)
        bind_socket.setblocking(1)

        self.max_conn_num = MAX_CONNECTIONS
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

    def connection_listen(self, source_ip: int):
        (conn, addr) = self.connections[source_ip]
        logger.info(f"Running in background, listening for ip {source_ip}")
        logger.debug(f"{conn}, {addr}")
        while True:
            try:
                data_raw = conn.recv(PACKET_SIZE)
                if data_raw != b"":
                    data = pickle.loads(data_raw)
                    if isinstance(data, NetworkPacket):
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
                            logger.error(self.connections[destination_ip])
                            logger.error(
                                f"Packet from ip {source_ip} was not forwarded, host"
                                f" not found with ip {destination_ip}"
                            )
                        except Exception as e:
                            logger.error(
                                "Error when sending packet received from              "
                                f"                       ip {source_ip} to ip"
                                f" {destination_ip}: {e}"
                            )
                    except KeyError:  # This source_ip was not assigned yet
                        logger.error(
                            f"Packet from ip {source_ip} was not forwarded,            "
                            "                     host not found with ip"
                            f" {destination_ip}"
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
                    "Critical error when receiving packet                         on"
                    f" ip {source_ip}: {e}"
                )
                self.connections[source_ip] = None
                exit(1)


if __name__ == "__main__":
    router = Router()
