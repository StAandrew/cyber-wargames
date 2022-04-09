import pickle
import socket
import time

import gym
import numpy as np
from gym import envs, spaces
from stable_baselines3.common.callbacks import BaseCallback

from common import MyPacket

N_DISCRETE_ACTIONS = 2
N_CHANNELS = 2
HOST = socket.gethostname()
PORT = 5432
PACKETS_PER_ITERATION = 10
DEBUG = False


class NwDefAgent(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(NwDefAgent, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # self.observation_space = spaces.Box(low=0, high=5,
        #         shape=(N_CHANNELS,), dtype=np.float32)
        self.observation_space = spaces.Discrete(5)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.settimeout(10)
        self.server_socket.listen(1)
        self.conn, self.addr = self.server_socket.accept()
        if DEBUG:
            print("Connection from: " + str(self.addr))

    def step(self, action):
        self.step_num += 1
        if DEBUG:
            print(f"step {self.step_num}")
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

        # send reward
        send_time = time.time()
        send_data = {"reward": reward}
        try:
            self.conn.send(pickle.dumps(send_data))
        except BrokenPipeError:
            info = {"finished": True}
            return [np.array(0), 0, False, info]
        except Exception as e:
            print("Exception when sending reward: ", e)
            exit(1)
        self.total_reward += reward
        if DEBUG:
            print("sent reward")

        # receive new packet
        try:
            recv_data = self.conn.recv(8196)
        except ConnectionResetError:
            if DEBUG:
                print("attacker is done")
            info = {"finished": True}
            return [np.array(0), 0, False, info]
        except Exception as e:
            if DEBUG:
                print("Exception when receiving packet: ", e)
            exit(1)
        if not recv_data:
            if DEBUG:
                print("EXCEPTION, finished")
            info = {"finished": True}
            return [np.array(0), 0, False, info]
        try:
            pkt = pickle.loads(recv_data)
        except Exception as e:
            if DEBUG:
                print("1 EXCEPTION ", e)

        if DEBUG:
            print("got packet")
        receive_time = time.time_ns()

        # print("src_ip: ", pkt.src_ip)
        observation = [pkt.src_ip]
        observation = np.array(observation)
        info = {"finished": False}

        self.prev_pkt = pkt
        return observation, reward, self.done, info

    def reset(self):
        self.step_num = 0
        if DEBUG:
            print("reset")

        self.done = False
        self.total_reward = 0

        self.correct_pkts = 0
        self.incorrect_pkts = 0

        if DEBUG:
            print("waiting for packets")
        try:
            data = self.conn.recv(8196)  # 4096
        except ConnectionResetError:
            if DEBUG:
                print("attacker is done")
            info = {"finished": True}
            return [np.array(0)]
        if not data:
            if DEBUG:
                print("attacker is done")
            info = {"finished": True}
            return [np.array(0)]

        if DEBUG:
            print("pickle data: ", data)

        try:
            pkt = pickle.loads(data)
        except Exception as e:
            if DEBUG:
                print("2 EXCEPTION: ", e)
            exit(1)

        if DEBUG:
            print("got packet")

        observation = [pkt.src_ip]
        observation = np.array(observation)

        self.prev_pkt = pkt
        return observation

    def render(self):
        print("---- episode ----")
        # print(self.pkt.str())

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
                if DEBUG:
                    print("training finished")
                continue_training = False
        except:
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
