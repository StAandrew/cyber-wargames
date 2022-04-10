import pickle
import random
import socket
import time

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback

from common import MyPacket, send_and_receive
from config import HOST, PORT, logger
from bg_traffic import BackgroundTraffic

N_DISCRETE_ACTIONS = 2
N_DISCRETE_SPACES = 2
PACKETS_PER_ITERATION = 10
DEBUG = False


class AttackingAgent(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(AttackingAgent, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1, high=10, shape=(N_DISCRETE_SPACES,), dtype=np.float32
        )

        self.client_socket = socket.socket()
        self.client_socket.settimeout(10)
        self.client_socket.connect((HOST, PORT))
        logger.debug("Attacker connected")

        self.background_traffic = BackgroundTraffic()
        logger.debug("Background connected")

    def step(self, action):
        logger.debug("---step---")

        if self.training_finished:
            logger.debug("training finished - beginning")
            info = {"finished": True}
            return [np.array([0, 0]), 0, False, info]

        atk_packet = getSampleAttackerPacket()
        atk_packet.src_ip = action

        sent_time = time.time_ns()
        recv_data, self.training_finished = send_and_receive(
            self.client_socket, atk_packet, self.rtt
        )
        self.past_rtt_list.append((time.time_ns() - sent_time) / 1000000000)
        self.rtt = np.average(self.past_rtt_list)
        logger.debug("rtt: ", self.rtt)
        logger.debug("sent atk ", self.step_num)
        self.step_num += 1

        if self.training_finished:
            logger.debug("training finished - var")
            info = {"finished": True}
            return [np.array([0, 0]), 0, False, info]

        try:
            recv_data = pickle.loads(recv_data)
        except EOFError as e:
            logger.debug(f"Error when processing received packet, training finished: {e}")
            info = {"finished": True}
            return [np.array([0, 0]), 0, False, info]
        reward = -1 * np.int32((recv_data.get("reward")))
        logger.debug("got reward: ", reward)

        if reward > 0:
            self.correct_pkts += 1
        else:
            self.incorrect_pkts -= 1

        if self.correct_pkts + self.incorrect_pkts > PACKETS_PER_ITERATION:
            self.done = True
        else:
            self.done = False

        observation = [reward, atk_packet.src_ip]
        observation = np.array(observation, dtype=np.float32)
        info = {"finished": False}

        return observation, reward, self.done, info

    def reset(self):
        logger.debug("---reset---")
        self.background_traffic.reset()

        self.step_num = 1
        self.correct_pkts = 0
        self.incorrect_pkts = 0
        self.done = False
        self.past_rtt_list = []
        self.rtt = 0.1  # seconds
        self.training_finished = False

        random_packet = getSampleAttackerPacket()
        sent_time = time.time_ns()
        recv_data, self.training_finished = send_and_receive(
            self.client_socket, random_packet, self.rtt
        )
        self.past_rtt_list.append((time.time_ns() - sent_time) / 1000000000)
        self.rtt = np.average(self.past_rtt_list)
        logger.debug("rtt: ", self.rtt)

        if self.training_finished:
            logger.debug("training finished in reset")
            info = {"finished": True}
            return np.array([0, 0])

        recv_data = pickle.loads(recv_data)
        reward = -1 * np.int32((recv_data.get("reward")))
        logger.debug("got reward: ", reward)

        self.step_num += 1
        self.client_socket.settimeout(10)  # TODO need to change
        observation = [reward, random_packet.src_ip]
        observation = np.array(observation, dtype=np.float32)
        return observation

    def render(self):
        logger.debug("---- episode ----")

    def close(self):
        logger.debug("close")
        # self.client_socket.shutdown(socket.SHUT_RDWR)
        self.client_socket.close()


"""
IP:
 -> defender
0 -> background
1 -> attacker
"""

def getSampleAttackerPacket():
    pkt = MyPacket(size=4, src_ip=1, dst_ip=0, true_source=1)
    return pkt


# if __name__ == "__main__":
#     client_socket = socket.socket()
#     client_socket.connect((ATK_HOST, ATK_PORT))
#     logger.debug("Connected")


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
        except:
            raise Exception("Error in callback! No finished info found")
        return continue_training
