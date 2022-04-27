"""Train Defender

Must be used in conjunction with Router (router.py) and Train Attacker (train_def.py).

Model type and information, and training information are specified as constants.
"""
import os
import time
from pathlib import Path
import threading

from stable_baselines3 import A2C, DQN, PPO

from env_def import CustomCallback, DefendingAgent
from config import models_dir, log_dir, logger, INITIAL_RTT, MAX_RETRIES
from bg_traffic import BackgroundTraffic


MODEL_TYPE = "DQN"
TIMESTEPS = 1000000000
SAVE_EVERY_TIMESTEPS = 100000
EXPLORATION_INITIAL = 1.0
EXPLORATION_FINAL = 0.3


logger.info("Initialising")
model_dir = Path(models_dir, f"network-def-{MODEL_TYPE}-{int(time.time())}")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    logger.debug("Model directory created")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    logger.debug("Log directory created")

log_time = int(time.time())
start_time = time.time()

# background traffic stared on a separate thread
background_traffic = BackgroundTraffic()
background_traffic.reset()
background_traffic_thread = threading.Thread(
    target=background_traffic.run, args=()
).start()

env = DefendingAgent()
custom_callback = CustomCallback()

# exploration increased from default
model = DQN(
    "MlpPolicy",
    env,
    verbose=0,
    exploration_initial_eps=EXPLORATION_INITIAL,
    exploration_fraction=0,
    exploration_final_eps=EXPLORATION_FINAL,
    tensorboard_log=log_dir,
)

logger.info("Starting training")
for i in range(1, int(TIMESTEPS / SAVE_EVERY_TIMESTEPS) + 1):
    model.learn(
        total_timesteps=SAVE_EVERY_TIMESTEPS,
        callback=custom_callback,
        reset_num_timesteps=False,
        tb_log_name=f"network-def-{MODEL_TYPE}-{log_time}",
    )
    logger.info(f"Saved {SAVE_EVERY_TIMESTEPS*i}")
    model.save(Path(model_dir, str(SAVE_EVERY_TIMESTEPS * i)))

end_time = time.time()
took_time = end_time - start_time
background_traffic.close()
env.close()
logger.info(f"Training finished. Took {took_time} seconds")
