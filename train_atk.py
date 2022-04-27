"""Train Attacker

Must be used in conjunction with Router (router.py) and Train Defender (train_def.py).

Model type and information, and training information are specified as constants.
"""
import os
import time
from pathlib import Path
import numpy as np

from stable_baselines3 import DDPG, DQN, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise

from env_atk import CustomCallback, AttackingAgent
from config import models_dir, log_dir, logger, MAX_RETRIES, INITIAL_RTT


MODEL_TYPE = "TD3"
TIMESTEPS = 200000000
SAVE_EVERY_TIMESTEPS = 20000
ACTION_NOISE_SIGMA = 0.1


logger.info("Initialising")
model_dir = Path(models_dir, f"network-atk-{MODEL_TYPE}-{int(time.time())}")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    logger.debug("Model directory created")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    logger.debug("Log directory created")

log_time = int(time.time())
start_time = time.time()

env = AttackingAgent()
custom_callback = CustomCallback()

# action noise added to increase exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=ACTION_NOISE_SIGMA * np.ones(n_actions)
)
model = TD3(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=log_dir,
)

logger.info("Starting training")
for i in range(1, int(TIMESTEPS / SAVE_EVERY_TIMESTEPS) + 1):
    model.learn(
        total_timesteps=SAVE_EVERY_TIMESTEPS,
        callback=custom_callback,
        reset_num_timesteps=False,
        tb_log_name=f"network-atk-{MODEL_TYPE}-{log_time}",
    )
    logger.info(f"Saved {SAVE_EVERY_TIMESTEPS*i}")
    model.save(Path(model_dir, str(SAVE_EVERY_TIMESTEPS * i)))

end_time = time.time()
took_time = end_time - start_time
env.close()
logger.info(f"Training finished. Took {took_time} seconds")
