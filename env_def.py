"""Defender Environment

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
import socket
import time

import gym
import numpy as np
from gym import envs, spaces
from stable_baselines3.common.callbacks import BaseCallback
from socket import timeout

from common import NetworkPacket
from config import HOST, PORT, logger, PACKET_SIZE

N_DISCRETE_ACTIONS = 2
N_CHANNELS = 2
PACKETS_PER_ITERATION = 300
MY_IP = 1
DTYPE = np.int32


class DefendingAgent(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(DefendingAgent, self).__init__()

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=5000, shape=(N_CHANNELS,), dtype=DTYPE
        )

        # connect to router and send empty packet with self ip
        self.socket = socket.socket()
        self.socket.settimeout(10)
        self.socket.connect((HOST, PORT))
        self.socket.send(
            pickle.dumps(
                NetworkPacket(0, 0, 0, true_source_ip=MY_IP, true_destination_ip=0)
            )
        )
        logger.debug("Defender connected")

    def step(self, action):
        # used as a substitution for callback when not training
        if bool(self.info["finished"]):
            return self.training_finished_on_step()

        self.step_num += 1
        if self.prev_pkt.true_source_ip == 3:
            self.bg_pkts += 1
        elif self.prev_pkt.true_source_ip == 2:
            self.atk_pkts += 1
        atk_def_ratio = 100 * self.atk_pkts / (self.atk_pkts + self.bg_pkts)
        logger.debug(f"Step {self.step_num}. Atk packets {atk_def_ratio:.0f}%")
        """
        Discrete actions:
        0 -> allow
        1 -> deny
        """
        if action == 0:
            self.prev_pkt.destination_ip = 10
        elif action == 1:
            self.prev_pkt.destination_ip = 11

        reward = 0
        # bg, pass, true positive
        if self.prev_pkt.true_source_ip == 3 and action == 0:
            reward = 1
            self.correct_pkts += 1
        # atk, discard, true negative
        elif self.prev_pkt.true_source_ip == 2 and action == 1:
            reward = 1  # 0
            self.correct_pkts += 1
        # bg, discard, false negative
        elif self.prev_pkt.true_source_ip == 3 and action == 1:
            reward = -1
            self.incorrect_pkts += 1
        # atk, pass, false positive
        elif self.prev_pkt.true_source_ip == 2 and action == 0:
            reward = -1  # -2
            self.incorrect_pkts += 1

        # mark the episode as done
        if (self.correct_pkts + self.incorrect_pkts) > PACKETS_PER_ITERATION:
            self.done = True
        else:
            self.done = False

        # send reward packet
        send_data = pickle.dumps(
            {
                "reward": reward,
                "true_source_ip": MY_IP,
                "true_destination_ip": self.prev_pkt.true_source_ip,
            }
        )
        try:
            self.socket.send(send_data)
        except BrokenPipeError as e:
            logger.error(f"Error when sending packet, training finished: {e}")
            return self.training_finished_on_step()
        # stop training in case of critical unexpected error
        except Exception as e:
            logger.critical(f"Critical error when sending packet: {e}")
            raise e  # stop training in case of critical unexpected error
        self.total_reward += reward
        logger.debug("sent reward")

        # receive new packet
        self.start_time = time.time()
        try:
            recv_data = self.socket.recv(PACKET_SIZE)
            self.time_delta = 10000 * round((time.time() - self.start_time), 4)
            logger.debug(f"time delta: {self.time_delta}")
        except ConnectionResetError:
            logger.debug("Attacker is done")
            return self.training_finished_on_step()
        except (TimeoutError, timeout) as e:
            logger.error(f"Timed out when receiving packet, training finished.")
            return self.training_finished_on_step()
        except Exception as e:
            logger.critical(f"Critical error when receiving packet: {e}")
            raise e  # stop training in case of critical unexpected error

        # check if packet is not empty
        if not recv_data:
            logger.error(f"Error when receiving packet, training finished.")
            return self.training_finished_on_step()
        try:
            pkt = pickle.loads(recv_data)
        except Exception as e:
            logger.error(f"Unexpected exception when loading new packet: {e}")

        # check if stop signal was received
        # used as a substitution for callback when not training
        try:
            if not isinstance(pkt, NetworkPacket):
                if dict(pkt).get("stop"):
                    logger.debug(f"Received stop signal, training finished")
                    return self.training_finished_on_step()
        # packet is normal, continue as usual
        except AttributeError:
            pass

        logger.debug(f"Received packet from true ip {pkt.true_source_ip}")
        observation = [pkt.source_ip, self.time_delta]
        observation = np.array(observation)
        self.info = {"finished": False}

        self.prev_pkt = pkt
        return observation, reward, self.done, self.info

    def reset(self):
        logger.debug("Reset")
        self.step_num = 0

        self.done = False
        self.info = {"finished": False}
        self.total_reward = 0

        self.correct_pkts = 0
        self.incorrect_pkts = 0
        self.atk_pkts = 0
        self.bg_pkts = 0

        self.start_time = time.time()
        logger.debug("Waiting for packets")
        try:
            data = self.socket.recv(PACKET_SIZE)
            self.time_delta = 10000 * round((time.time() - self.start_time), 4)
            logger.debug(f"time delta: {self.time_delta}")
        except ConnectionResetError as e:
            logger.debug("Attacker is done: {e}")
            return self.training_finished_on_reset()
        if not data:
            logger.debug("Attacker is done. No data.")
            return self.training_finished_on_reset()

        try:
            pkt = pickle.loads(data)
        except Exception as e:
            logger.critical(
                f"Critical unexpected exception when loading packet data: {e}"
            )
            raise e

        logger.debug("Got packet")

        # check if stop signal was received
        # used as a substitution for callback when not training
        try:
            if not isinstance(pkt, NetworkPacket):
                if dict(pkt).get("stop"):
                    logger.debug(f"Received stop signal, training finished.")
                    self.info = {"finished": True}
                    return np.array([0, 0])
        # packet is normal, continue as usual
        except AttributeError:
            pass

        observation = [pkt.source_ip, self.time_delta]
        observation = np.array(observation)

        self.prev_pkt = pkt
        return observation

    def render(self):
        logger.debug("---- episode ----")
        # logger.debug(self.pkt.str())

    def close(self):
        self.socket.close()

    def training_finished_on_step(self):
        self.info = {"finished": True}
        return [np.array([0, 0]), 0, False, self.info]

    def training_finished_on_reset(self):
        self.info = {"finished": True}
        return np.array([0, 0])


"""
Custom callback used to signal that training is finished.
Training is deemed to be finished when the Attacker cannot be reached.
"""


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
