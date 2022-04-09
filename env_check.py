from stable_baselines3.common.env_checker import check_env
from env_def import NwDefAgent
from env_atk import NwAtkAgent
import asyncio
from threading import Thread


async def def_check():
    env_def = NwDefAgent()
    await check_env(env_def)


async def atk_check():
    env_atk = NwAtkAgent()
    await check_env(env_atk)


async def main():
    await asyncio.gather(def_check(), atk_check())


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

episodes = 50

# for episode in range(episodes):
#     print("episode {}".format(episode))
#     done = False
#     obs = env.reset()
#     while not done:
#         random_action = env.action_space.sample()
#         print("action", random_action)
#         obs, reward, done, info = env.step(random_action)
#         print("reward", reward)
