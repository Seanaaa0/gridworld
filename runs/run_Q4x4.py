import gym
from gym.envs.registration import register
from grid_mdp2 import GridEnv4x4
import random
import numpy as np
import time

# 註冊環境
register(
    id='GridWorld4x4-v0',
    entry_point='grid_mdp2:GridEnv4x4',
)

env = gym.make('GridWorld4x4-v0')
actions = env.getAction()
num_states = len(env.getStates())
num_actions = len(actions)
q_table = np.zeros((num_states + 1, num_actions))  # 狀態從 1 開始

# Q-learning 參數
alpha = 0.1
gamma = 0.9
initial_epsilon = 1.0


episodes = 1000

# 訓練階段
for episode in range(episodes):
    # 保證不是 goal 起始
    epsilon = max(0.1, initial_epsilon - episode / 400)  # 從 1 漸漸降到 0.1

    while True:
        state = env.reset()
        if env.terminate_states.get(state) != 'goal':
            break

    done = False
    steps = 0
    while not done and steps < 50:
        if random.uniform(0, 1) < epsilon:
            a_index = random.randint(0, num_actions - 1)
        else:
            a_index = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(a_index)
        old_value = q_table[state][a_index]
        next_max = np.max(q_table[next_state])

        # Q-learning 更新
        q_table[state][a_index] = old_value + alpha * \
            (reward + gamma * next_max - old_value)
        state = next_state
        steps += 1

# 測試階段
print("======= 測試策略 =======")
state = env.reset()
while env.terminate_states.get(state) == 'goal':
    state = env.reset()

done = False
attempts = 0
env.render()

while not done and attempts < 25:
    a_index = np.argmax(q_table[state])
    obs, reward, done, _ = env.step(a_index)

    print(
        f"狀態 {state} + 動作 {actions[a_index]} → 狀態 {obs}，獎勵: {reward}, 結束: {done}")
    state = obs
    attempts += 1
    env.render()
    time.sleep(0.5)

env.close()
