import gym
from gym.envs.registration import register
from grid_mdp import GridEnv
import random
import numpy as np
import time

register(
    id='GridWorld-v0',
    entry_point='grid_mdp:GridEnv',
)

env = gym.make('GridWorld-v0')

actions = env.getAction()
num_states = 9
num_actions = len(actions)
q_table = np.zeros((num_states, num_actions))

alpha = 0.1      # 學習率
gamma = 0.9      # 折扣率
epsilon = 0.2    # 探索率
episodes = 500   # 迭代次數

for episode in range(episodes):
    while True:
        state = env.reset()
        if isinstance(state, tuple):  # gym 0.26+ 格式
            state = state[0]
        if state not in env.terminate_states:
            break
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            a_index = random.randint(0, num_actions - 1)
        else:
            a_index = np.argmax(q_table[state])

        obs, reward, done, _ = env.step(a_index)

        # Q-learning 更新
        old_value = q_table[state][a_index]
        next_max = np.max(q_table[obs])
        q_table[state][a_index] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = obs

# 使用訓練完的 Q-table 測試一次
print("======= 測試策略 =======")
state = env.reset()
if isinstance(state, tuple):
    state = state[0]
done = False
env.render()

while not done:
    a_index = np.argmax(q_table[state])
    obs, reward, done, _ = env.step(a_index)

    print(f"狀態 {state} + 動作 {actions[a_index]} → 狀態 {obs}，獎勵: {reward}, 結束: {done}")
    state = obs
    env.render()
    time.sleep(0.5)

env.close()
