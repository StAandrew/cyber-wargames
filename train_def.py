import gym
import time, os
from stable_baselines3 import PPO, DQN
from env_def import CustomCallback, NwDefAgent
from common import MyPacket

model_type = "DQN"
models_dir = f"models/network-def-{model_type}-{int(time.time())}"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

timesteps = 400000
save_every_timesteps = 10000
log_time = int(time.time())
start_time = time.time()
env = NwDefAgent()
custom_callback = CustomCallback()

model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

for i in range(1, int(timesteps / save_every_timesteps) + 1):
    model.learn(
        total_timesteps=save_every_timesteps,
        callback=custom_callback,
        reset_num_timesteps=False,
        tb_log_name=f"network-def-{model_type}-{log_time}",
    )
    model.save(f"{models_dir}/{save_every_timesteps*i}")

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
