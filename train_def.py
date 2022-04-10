import os
import time
from pathlib import Path

import gym
from stable_baselines3 import DQN, PPO

from common import MyPacket
from env_def import CustomCallback, NwDefAgent
from config import models_dir, log_dir

model_type = "DQN"
model_dir = Path(models_dir, f"network-def-{model_type}-{int(time.time())}")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timesteps = 400000
save_every_timesteps = 10000
log_time = int(time.time())
start_time = time.time()
env = NwDefAgent()
custom_callback = CustomCallback()

model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=log_dir)
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

for i in range(1, int(timesteps / save_every_timesteps) + 1):
    model.learn(
        total_timesteps=save_every_timesteps,
        callback=custom_callback,
        reset_num_timesteps=False,
        tb_log_name=f"network-def-{model_type}-{log_time}",
    )
    model.save(Path(model_dir, save_every_timesteps * i))

env.close()
print("-----model training done-----")
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
