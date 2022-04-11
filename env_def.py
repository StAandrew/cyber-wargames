import pickle
import socket
import time

import gym
import numpy as np
from gym import envs, spaces
from stable_baselines3.common.callbacks import BaseCallback

from common import MyPacket
from config import HOST, PORT, logger, PACKET_SIZE

N_DISCRETE_ACTIONS = 2
N_CHANNELS = 2
PACKETS_PER_ITERATION = 10
MY_IP = 1


class DefendingAgent(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(DefendingAgent, self).__init__()

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=10,
                shape=(N_CHANNELS,), dtype=np.float32)
        # self.observation_space = spaces.Discrete(5)

        self.socket = socket.socket()
        self.socket.settimeout(10)
        self.socket.connect((HOST, PORT))
        self.socket.send(
            pickle.dumps(MyPacket(0, 0, 0, true_source_ip=MY_IP, true_destination_ip=0))
        )
        logger.debug("Defender connected")

    def step(self, action):
        self.step_num += 1
        logger.debug(f"Step {self.step_num}")
        """
        0 -> pass
        1 -> deny
        """
        if action == 0:
            self.prev_pkt.destination_ip = 10
            # self.accepted_pkts.append(pkt)
        elif action == 1:
            self.prev_pkt.destination_ip = 11
            # self.discarded_pkts.append(pkt)

        reward = 0
        if self.prev_pkt.true_source_ip == 3 and action == 0:  # bg, pass
            reward = 1
            self.correct_pkts += 1
        elif self.prev_pkt.true_source_ip == 2 and action == 1:  # atk, discard
            reward = 1
            self.correct_pkts += 1
        elif self.prev_pkt.true_source_ip == 3 and action == 1:  # bg, discard
            reward = -1
            self.incorrect_pkts += 1
        elif self.prev_pkt.true_source_ip == 2 and action == 0:  # atk, pass
            reward = -1
            self.incorrect_pkts += 1

        if self.correct_pkts + self.incorrect_pkts > PACKETS_PER_ITERATION:
            self.done = True
        else:
            self.done = False

        # send reward packet
        send_time = time.time()
        # feedback_packet = MyPacket(0, 0, 0, MY_IP, self.prev_pkt.true_source_ip)
        # feedback_packet.data = {"reward": reward}
        # send_data = pickle.dumps(feedback_packet)
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
            logger.error(f"Error when sending packet, episode finished: {e}")
            info = {"finished": True}
            return [np.array(0), 0, False, info]
        except Exception as e:
            logger.critical(f"Critical error when sending packet: {e}")
            exit(1)
        self.total_reward += reward
        logger.debug("sent reward")

        self.start_time = time.time()
        # receive new packet
        try:
            recv_data = self.socket.recv(PACKET_SIZE)
            self.time_delta = time.time() - self.start_time
            logger.debug(f"time delta: {self.time_delta}")
        except ConnectionResetError:
            logger.debug("Attacker is done")
            info = {"finished": True}
            return [np.array(0), 0, False, info]
        except TimeoutError as e:
            logger.error(f"Timed out when receiving packet, episode finished.")
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

        logger.debug("Received packet")

        # print("src_ip: ", pkt.src_ip)
        observation = [pkt.source_ip, self.time_delta]
        observation = np.array(observation)
        info = {"finished": False}

        self.prev_pkt = pkt
        return observation, reward, self.done, info

    def reset(self):
        self.step_num = 0
        logger.debug("Reset")

        self.done = False
        self.total_reward = 0

        self.correct_pkts = 0
        self.incorrect_pkts = 0

        self.start_time = time.time()
        logger.debug("Waiting for packets")
        try:
            data = self.socket.recv(PACKET_SIZE)
            self.time_delta = time.time() - self.start_time
            logger.debug(f"time delta: {self.time_delta}")
        except ConnectionResetError as e:
            logger.debug("Attacker is done: {e}")
            info = {"finished": True}
            return [np.array(0)]
        if not data:
            logger.debug("Attacker is done. No data.")
            info = {"finished": True}
            return [np.array(0)]

        try:
            pkt = pickle.loads(data)
        except Exception as e:
            logger.debug(f"Unexpected exception when loading packet data: {e}")
            exit(1)

        logger.debug("Got packet")

        observation = [pkt.source_ip, self.time_delta]
        observation = np.array(observation)

        self.prev_pkt = pkt
        return observation

    def render(self):
        logger.debug("---- episode ----")
        # logger.debug(self.pkt.str())

    def close(self):
        pass
        # self.server_socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()


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
