import os
import time
from pathlib import Path
import threading

from stable_baselines3 import DQN

from env_def import CustomCallback, DefendingAgent
from config import models_dir, log_dir, logger
from bg_traffic import BackgroundTraffic


logger.info("Initialising")
model_type = "DQN"
model_dir = Path(models_dir, f"network-def-{model_type}-{int(time.time())}")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    logger.debug("Model directory created")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    logger.debug("Log directory created")

timesteps = 40
save_every_timesteps = 10
log_time = int(time.time())
start_time = time.time()

background_traffic = BackgroundTraffic()
background_traffic.reset()
background_traffic_thread = threading.Thread(target=background_traffic.run, args=()).start()

env = DefendingAgent()
custom_callback = CustomCallback()
model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=log_dir)

logger.info("Starting training")
for i in range(1, int(timesteps / save_every_timesteps) + 1):
    model.learn(
        total_timesteps=save_every_timesteps,
        callback=custom_callback,
        reset_num_timesteps=False,
        tb_log_name=f"network-def-{model_type}-{log_time}",
    )
    logger.info(f"Saved {save_every_timesteps*i}")
    model.save(Path(model_dir, str(save_every_timesteps * i)))

background_traffic.close()
env.close()
logger.info("Training done")
# time.sleep()

# env = NwDefAgent()
# correct = 0
# total = 20
# for i in range(total):
#     obs = env.reset()
#     action, _verbose = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     print(f"obs: {obs}, action: {action}, reward: {reward}")
#     if reward == 1:
#         correct += 1
# end_time = time.time()
# print(f"timesteps: {timesteps}, correct {(100*correct/total):.2f}%, took {(end_time-start_time):.0f}s")
