"""Play Attacker

This script allows measuring performance of Attacker. 

Must be used in conjunction with Router (router.py), Play Defender (play_def.py) and 
optionally Background Traffic (bg_traffic.py).
"""
import os
import pathlib
import time

from stable_baselines3 import DDPG, DQN, TD3, SAC

from env_atk import AttackingAgent
from config import models_dir, log_dir, logger


file_name = "60000"
file_extention = ".zip"
starts_with = "network-atk-TD3"

found = False
for dir_name in os.listdir(models_dir):
    if dir_name.startswith(starts_with) and not found:
        found = True
        model_path = pathlib.Path(models_dir, dir_name, f"{file_name}{file_extention}")
        logger.debug(f"path: {model_path}")
        if not os.path.exists(model_path):
            found = False
if not found:
    logger.error("no file/dir found")
    exit(1)

env = AttackingAgent()
env.reset()
model = TD3.load(model_path, env=env)

episodes = 2
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
        logger.debug(finished)
        logger.debug(f"obs: {obs}, action: {action}, reward: {reward}")
        if reward == 1:
            correct += 1
        total += 1
env.send_finished_signal()
end_time = time.time()
env.close()
logger.info(
    f"episodes: {episodes}, correct {(100*correct/total):.2f}%, took"
    f" {(end_time-start_time):.0f}s"
)
