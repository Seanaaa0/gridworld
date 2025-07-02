import gym
from gym.envs.registration import register
from grid_mdp import GridEnv
import random
import time  # ⬅️ 加這行：為了 sleep

register(
    id='GridWorld-v0',
    entry_point='grid_mdp:GridEnv',
)

env = gym.make('GridWorld-v0', render_mode="human")

# 每一輪最大步數
MAX_STEPS = 10
# 要跑幾回 episode
EPISODES = 5

for episode in range(EPISODES):
    print(f"======= Episode {episode + 1} =======")
    state = env.reset()
    done = False
    for step in range(MAX_STEPS):
        env.render()
        time.sleep(0.5)  # ⬅️ 每步停 0.5 秒，幫助視覺辨識
        action = env.getAction()
        a_index = random.randint(0, len(action) - 1)
        obs, reward, done, _ = env.step(a_index)
        print(f"狀態 {state} + 動作 {action[a_index]} → 狀態 {obs}，獎勵: {reward}, 結束: {done}")
        state = obs
        if done:
            break

env.close()
