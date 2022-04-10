import pickle
import socket
import time

import gym
import numpy as np
from gym import envs, spaces
from stable_baselines3.common.callbacks import BaseCallback

from common import MyPacket
from config import HOST, PORT, logger

N_DISCRETE_ACTIONS = 2
N_CHANNELS = 2
PACKETS_PER_ITERATION = 10


class DefendingAgent(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(DefendingAgent, self).__init__()

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # self.observation_space = spaces.Box(low=0, high=5,
        #         shape=(N_CHANNELS,), dtype=np.float32)
        self.observation_space = spaces.Discrete(5)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.settimeout(10)

        logger.debug("Ready to accept new connections")
        self.server_socket.listen(1)
        self.conn, self.addr = self.server_socket.accept()
        logger.debug("Connection from: " + str(self.addr))

        self.server_socket.listen(1)
        self.bg_conn, self.bg_addr = self.server_socket.accept()
        logger.debug("BG connection from: " + str(self.addr))

    def step(self, action):
        self.step_num += 1
        logger.debug(f"step {self.step_num}")
        """
        0 -> pass
        1 -> deny
        """
        if action == 0:
            self.prev_pkt.dst_ip = 4
            # self.accepted_pkts.append(pkt)
        elif action == 1:
            self.prev_pkt.dst_ip = 5
            # self.discarded_pkts.append(pkt)

        reward = 0
        if self.prev_pkt.true_source == 0 and action == 0:  # bg, pass
            reward = 1
            self.correct_pkts += 1
        elif self.prev_pkt.true_source == 1 and action == 1:  # atk, discard
            reward = 1
            self.correct_pkts += 1
        elif self.prev_pkt.true_source == 0 and action == 1:  # bg, discard
            reward = -1
            self.incorrect_pkts += 1
        elif self.prev_pkt.true_source == 1 and action == 0:  # atk, pass
            reward = -1
            self.incorrect_pkts += 1

        if self.correct_pkts + self.incorrect_pkts > PACKETS_PER_ITERATION:
            self.done = True
        else:
            self.done = False

        # send reward packet
        send_time = time.time()
        send_data = {"reward": reward}
        try:
            self.conn.send(pickle.dumps(send_data))
        except BrokenPipeError as e:
            logger.error(f"Error when sending packet, episode finished: {e}")
            info = {"finished": True}
            return [np.array(0), 0, False, info]
        except Exception as e:
            logger.critical(f"Critical error when sending packet: {e}")
            exit(1)
        self.total_reward += reward
        logger.debug("sent reward")

        # receive new packet
        try:
            recv_data = self.conn.recv(8196)
        except ConnectionResetError:
            logger.debug("attacker is done")
            info = {"finished": True}
            return [np.array(0), 0, False, info]
        except Exception as e:
            logger.critical(f"Critical error when receiving packet: {e}")
            exit(1)
        if not recv_data:
            logger.error(f"Error when receiving packet, episode finished.")
            info = {"finished": True}
            return [np.array(0), 0, False, info]
        try:
            pkt = pickle.loads(recv_data)
        except Exception as e:
            logger.error(f"Unexpected exception when loading new packet: {e}")

        logger.debug("received packet")
        receive_time = time.time_ns()

        # print("src_ip: ", pkt.src_ip)
        observation = [pkt.src_ip]
        observation = np.array(observation)
        info = {"finished": False}

        self.prev_pkt = pkt
        return observation, reward, self.done, info

    def reset(self):
        self.step_num = 0
        logger.debug("reset")

        self.done = False
        self.total_reward = 0

        self.correct_pkts = 0
        self.incorrect_pkts = 0

        logger.debug("waiting for packets")
        try:
            data = self.conn.recv(8196)  # 4096
        except ConnectionResetError as e:
            logger.debug("Attacker is done: {e}")
            info = {"finished": True}
            return [np.array(0)]
        if not data:
            logger.debug("Attacker is done. ")
            info = {"finished": True}
            return [np.array(0)]

        logger.debug("pickle data: ", data)

        try:
            pkt = pickle.loads(data)
        except Exception as e:
            logger.debug(f"Unexpected exception when loading packet data: {e}")
            exit(1)

        logger.debug("got packet")

        observation = [pkt.src_ip]
        observation = np.array(observation)

        self.prev_pkt = pkt
        return observation

    def render(self):
        logger.debug("---- episode ----")
        logger.debug(self.pkt.str())

    def close(self):
        pass
        # self.server_socket.shutdown(socket.SHUT_RDWR)
        self.server_socket.close()


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        continue_training = True
        try:
            finished_bool = bool(self.locals["infos"][0].get("finished"))
            if finished_bool:
                logger.debug("training finished")
                continue_training = False
        except Exception as e:
            logger.error("Error in callback! No finished info found {e}")
            raise Exception("Error in callback! No finished info found")
        return continue_training


# def translate_action(action):
#     if action == 0:
#         return "allow"
#     elif action == 1:
#         return "deny"

# def translate_address(address):
#     if address == 1:
#         return "defender"
#     elif address == 2:
#         return "background"
#     elif address == 3:
#         return "attacker"
#     elif address == 4:
#         return "pass"
#     elif address == 5:
#         return "discard"
