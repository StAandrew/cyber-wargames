"""Attacker Enviroment

OpenAI Gym based environment for Defenging Agent.

N_DISCRETE_ACTIONS: int
    the number of discrete actions agent can take
N_CHANNELS: int
    the number of observation channels of the agent
PACKETS_PER_ITERATION: int
    the number of packets agent observes during a single training iteration
MY_IP: int
    IP address of the agent. For simplicity, kept as int value
DTYPE: 
    Numpy data type used by the agent. Typically, np.float32
"""
import pickle
import random
import socket
import time
import threading
from math import floor

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback

from common import NetworkPacket, send_and_receive
from config import HOST, PORT, logger, INITIAL_RTT
from bg_traffic import BackgroundTraffic

N_DISCRETE_ACTIONS = 2
N_DISCRETE_SPACES = 1
PACKETS_PER_ITERATION = 150
MY_IP = 2
DEFENDER_IP = 1
DEFAULT_PACKET_SIZE = 4
DTYPE = np.float32


class AttackingAgent(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(AttackingAgent, self).__init__()
        self.action_space = spaces.Box(
            low=0, high=500, shape=(N_DISCRETE_ACTIONS,), dtype=DTYPE
        )
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(N_DISCRETE_SPACES,), dtype=DTYPE
        )

        # connect to router and send empty packet with self ip
        self.client_socket = socket.socket()
        self.client_socket.settimeout(10)
        self.client_socket.connect((HOST, PORT))
        self.client_socket.send(
            pickle.dumps(
                NetworkPacket(0, 0, 0, true_source_ip=MY_IP, true_destination_ip=0)
            )
        )
        logger.debug("Attacker connected")

    def step(self, action):
        logger.debug(f"--- step {self.step_num} ---")
        self.step_num += 1

        if self.training_finished:
            logger.debug("Training finished.")
            return self.training_finished_on_step

        atk_packet = NetworkPacket(
            size=4,
            source_ip=MY_IP,
            destination_ip=DEFENDER_IP,
            true_source_ip=MY_IP,
            true_destination_ip=DEFENDER_IP,
        )

        """
        Continious actions:
        Action[0] -> source ip address of the packet
        Action[1] -> time to wait before sending the next packet
        """
        atk_packet.source_ip = floor(action[0])
        to_sleep = action[1] / 10000

        logger.debug(f"to sleep: {to_sleep}")
        time.sleep(to_sleep)

        sent_time = time.time_ns()
        recv_data, self.training_finished = send_and_receive(
            self.client_socket, atk_packet, self.rtt, __name__
        )
        self.past_rtt_list.append((time.time_ns() - sent_time) / 1000000000)
        self.rtt = np.average(self.past_rtt_list)
        logger.debug(f"rtt: {self.rtt}")

        if self.training_finished:
            logger.debug("training finished - var")
            return self.training_finished_on_step
        try:
            recv_data = pickle.loads(recv_data)
        except EOFError as e:
            logger.debug(
                f"Error when processing received packet, training finished: {e}"
            )
            return self.training_finished_on_step()
        reward = -1 * np.int32(int(recv_data.get("reward")))  # reward must be integer
        logger.debug(f"got reward: {reward}")

        if reward > 0:
            self.correct_pkts += 1
        else:
            self.incorrect_pkts += 1

        # mark the episode as done
        if (self.correct_pkts + self.incorrect_pkts) > PACKETS_PER_ITERATION:
            self.done = True
        else:
            self.done = False

        observation = [reward]
        observation = np.array(observation, dtype=DTYPE)
        logger.debug(f"Oberrvation: {observation}")

        info = {"finished": False}
        return observation, reward, self.done, info

    def reset(self):
        logger.debug("---reset---")

        self.step_num = 1
        self.correct_pkts = 0
        self.incorrect_pkts = 0
        self.done = False
        self.past_rtt_list = []
        self.rtt = INITIAL_RTT
        self.training_finished = False

        # send a random packet
        random_packet = NetworkPacket(
            size=4,
            source_ip=MY_IP,
            destination_ip=DEFENDER_IP,
            true_source_ip=MY_IP,
            true_destination_ip=DEFENDER_IP,
        )
        sent_time = time.time_ns()
        recv_data, self.training_finished = send_and_receive(
            self.client_socket, random_packet, self.rtt, __name__
        )
        self.past_rtt_list.append((time.time_ns() - sent_time) / 1000000000)
        self.rtt = np.average(self.past_rtt_list)
        logger.debug(f"rtt: {self.rtt}")

        if self.training_finished:
            logger.debug("episode finished on reset")
            return self.training_finished_on_reset()

        recv_data = pickle.loads(recv_data)
        reward = -1 * np.int32((recv_data.get("reward")))
        logger.debug(f"got reward: {reward}")

        self.step_num += 1
        self.client_socket.settimeout(10)  # TODO need to change
        observation = [reward]
        observation = np.array(observation, dtype=DTYPE)
        logger.debug(f"Oberrvation: {observation}")
        return observation

    def render(self):
        logger.debug("---- episode ----")

    def close(self):
        logger.debug("close")
        # self.client_socket.shutdown(socket.SHUT_RDWR)
        self.client_socket.close()

    def send_finished_signal(self):
        recv_data, self.training_finished = send_and_receive(
            self.client_socket,
            {"stop": True, "true_source_ip": 2, "true_destination_ip": 1},
            self.rtt,
            __name__,
        )

    def training_finished_on_step(self):
        self.info = {"finished": True}
        return [np.array([0, 0]), 0, False, self.info]

    def training_finished_on_reset(self):
        self.info = {"finished": True}
        return np.array([0, 0])


"""
Custom callback used to signal that training is finished.
Training is deemed to be finished when the Defender cannot be reached.
"""


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        continue_training = True
        try:
            finished_bool = bool(self.locals["infos"][0].get("finished"))
            if finished_bool:
                logger.debug("episode finished")
                continue_training = False
        except Exception as e:
            logger.critical(f"Critical callback exception: {e}")
            raise Exception("Error in callback! No finished info found")
        return continue_training
