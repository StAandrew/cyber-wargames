"""Play Defender

This script allows measuring performance of Defender. 

Must be used in conjunction with Router (router.py), Play Attacker (play_def.py) and 
optionally Background Traffic (bg_traffic.py).

File name, extention and directory name must be specified as variables. File contains
a trained model to be used.
"""
import os
import pathlib
import time
import threading

from stable_baselines3 import A2C, DQN, PPO

from env_def import DefendingAgent
from config import models_dir, log_dir, logger
from bg_traffic import BackgroundTraffic


FILE_NAME = "100000"
FILE_EXTENSION = ".zip"
STARTS_WITH = "network-def-DQN"

found = False
for dir_name in os.listdir(models_dir):
    if dir_name.startswith(STARTS_WITH) and not found:
        found = True
        model_path = pathlib.Path(models_dir, dir_name, f"{FILE_NAME}{FILE_EXTENSION}")
        logger.debug(f"path: {model_path}")
        if not os.path.exists(model_path):
            found = False

if not found:
    logger.error("no file/dir found")
    exit(1)

env = DefendingAgent()
env.reset()
model = DQN.load(model_path, env=env)

background_traffic = BackgroundTraffic()
background_traffic.reset()
background_traffic_thread = threading.Thread(
    target=background_traffic.run, args=()
).start()

episodes = 100
correct = 0
total = 0
finished = False

start_time = time.time()
for ep in range(episodes):
    if finished:
        break
    logger.debug("reset")
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
        logger.debug(f"obs: {obs}, action: {action}, reward: {reward}")
        if reward == 1:
            correct += 1
        total += 1

end_time = time.time()
background_traffic.close()
env.close()
logger.info(
    f"episodes: {episodes}, correct {(100*correct/total):.2f}%, took"
    f" {(end_time-start_time):.0f}s"
)
