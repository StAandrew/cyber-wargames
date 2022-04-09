import pickle
import random
import socket
import time

import gym
import numpy as np
from gym import spaces
from common import MyPacket
from stable_baselines3.common.callbacks import BaseCallback


N_DISCRETE_ACTIONS = 2
N_DISCRETE_SPACES = 2
ATK_HOST = socket.gethostname()
ATK_PORT = 5432
PACKETS_PER_ITERATION = 10
DEBUG = False


class NwAtkAgent(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(NwAtkAgent, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1, high=10, shape=(N_DISCRETE_SPACES,), dtype=np.float32
        )

        self.client_socket = socket.socket()
        self.client_socket.settimeout(10)
        self.client_socket.connect((ATK_HOST, ATK_PORT))
        if DEBUG:
            print("Connected")

    def step(self, action):
        if DEBUG:
            print("---step---")

        if self.training_finished:
            if DEBUG:
                print("training finished - beginning")
            info = {"finished": True}
            return [np.array([0, 0]), 0, False, info]

        if random.random() > 0.5:
            bg_packet = getBackgroundPacket()
            if DEBUG:
                print("sent bg ", self.step_num)
            self.step_num += 1
            recv_data = send_rcv_func(self.client_socket, bg_packet, self.rtt)

        atk_packet = getSampleAttackerPacket()
        atk_packet.src_ip = action

        sent_time = time.time_ns()
        recv_data, self.training_finished = send_rcv_func(
            self.client_socket, atk_packet, self.rtt
        )
        self.past_rtt_list.append((time.time_ns() - sent_time) / 1000000000)
        self.rtt = np.average(self.past_rtt_list)

        if DEBUG:
            print("rtt: ", self.rtt)
            print("sent atk ", self.step_num)
        self.step_num += 1

        if self.training_finished:
            if DEBUG:
                print("training finished - var")
            info = {"finished": True}
            return [np.array([0, 0]), 0, False, info]

        try:
            recv_data = pickle.loads(recv_data)
        except EOFError:
            if DEBUG:
                print("training finished - EOFError")
            info = {"finished": True}
            return [np.array([0, 0]), 0, False, info]
        reward = -1 * np.int32((recv_data.get("reward")))

        if DEBUG:
            print("got reward: ", reward)

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
        if DEBUG:
            print("---reset---")
        self.step_num = 1
        self.correct_pkts = 0
        self.incorrect_pkts = 0
        self.done = False
        self.past_rtt_list = []
        self.rtt = 0.1  # seconds
        self.training_finished = False

        random_packet = getSampleAttackerPacket()
        sent_time = time.time_ns()
        recv_data, self.training_finished = send_rcv_func(
            self.client_socket, random_packet, self.rtt
        )
        self.past_rtt_list.append((time.time_ns() - sent_time) / 1000000000)
        self.rtt = np.average(self.past_rtt_list)

        if DEBUG:
            print("rtt: ", self.rtt)

        if self.training_finished:
            info = {"finished": True}
            return np.array([0, 0])

        recv_data = pickle.loads(recv_data)
        reward = -1 * np.int32((recv_data.get("reward")))
        if DEBUG:
            print("got reward: ", reward)

        self.step_num += 1
        self.client_socket.settimeout(10)  # TODO need to change
        observation = [reward, random_packet.src_ip]
        observation = np.array(observation, dtype=np.float32)
        return observation

    def render(self):
        print("---- episode ----")

    def close(self):
        print("close")
        # self.client_socket.shutdown(socket.SHUT_RDWR)
        self.client_socket.close()


"""
IP:
 -> defender
0 -> background
1 -> attacker
"""


def send_rcv_func(socket, packet, rtt):
    recv_response = False
    training_finished = False
    if DEBUG:
        attempts = 0
    while not recv_response:
        if DEBUG:
            attempts += 1
            print("  send_rcv_func: attempt ", attempts)
        # send random packet
        try:
            socket.send(pickle.dumps(packet))
        except BrokenPipeError as e:
            if DEBUG:
                print("  send_rcv_func: training finished on send", e)
            training_finished = True
            return "", training_finished
        except Exception as e:
            print("  send_rcv_func: Exception! ", e)
            training_finished = True
            return "", training_finished
        if DEBUG:
            print("  send_rcv_func: packet sent")

        # receive response
        socket.settimeout(rtt)
        try:
            recv_data = socket.recv(4096)
            recv_response = True
            if DEBUG:
                print("  send_rcv_func: response received")
        except (EOFError, BrokenPipeError) as e:
            if DEBUG:
                print("  send_rcv_func: training finished on receive", e)
            training_finished = True
            return recv_data, training_finished
        except Exception:
            pass
    return recv_data, training_finished


def getSampleAttackerPacket():
    pkt = MyPacket(size=4, src_ip=1, dst_ip=0, true_source=1)
    return pkt


def getBackgroundPacket():
    pkt = MyPacket(size=4, src_ip=0, dst_ip=0, true_source=0)
    return pkt


# if __name__ == "__main__":
#     client_socket = socket.socket()
#     client_socket.connect((ATK_HOST, ATK_PORT))
#     print("Connected")


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        continue_training = True
        try:
            finished_bool = bool(self.locals["infos"][0].get("finished"))
            if finished_bool:
                print("training finished")
                continue_training = False
        except:
            raise Exception("Error in callback! No finished info found")
        return continue_training
