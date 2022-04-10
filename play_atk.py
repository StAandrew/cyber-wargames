import os
import pathlib
import time

import gym
from stable_baselines3 import A2C, DQN, PPO

from env_atk import AttackingAgent
from config import models_dir, log_dir

env = AttackingAgent()
env.reset()

file_name = "200000"
file_extention = ".zip"
starts_with = "network-atk-DQN-1648507701"

found = False
for dir_name in os.listdir(models_dir):
    if dir_name.startswith(starts_with) and not found:
        found = True
        model_path = pathlib.Path(models_dir, dir_name, file_name, file_extention)
        if not os.path.exists(model_path):
            found = False
if not found:
    print("no file/dir found")
    exit(1)

model = DQN.load(model_path, env=env)

episodes = 100
correct = 0
total = 0

for ep in range(episodes):
    print("reset")
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _verbose = model.predict(obs)
        obs, reward, done, info = env.step(action)
        finished = bool(info["finished"])
        if finished:
            done = True
            break
        print(finished)
        print(f"obs: {obs}, action: {action}, reward: {reward}")
        if reward == 1:
            correct += 1
        total += 1
print(f"episodes: {episodes}, correct {(100*correct/total):.2f}%")
env.close()
