import gym
from gym.envs.registration import register
from grid_mdp2 import GridEnv4x4
import numpy as np
import random
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

q_table = np.zeros((num_states + 1, num_actions))

# 超參數
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
episodes = 1000

# SARSA 訓練
for episode in range(episodes):
    state = env.reset()
    while env.terminate_states.get(state) == 'goal':
        state = env.reset()

    if random.uniform(0, 1) < epsilon:
        a_index = random.randint(0, num_actions - 1)
    else:
        a_index = np.argmax(q_table[state])

    done = False
    steps = 0

    while not done and steps < 50:
        next_state, reward, done, _ = env.step(a_index)

        if random.uniform(0, 1) < epsilon:
            next_a_index = random.randint(0, num_actions - 1)
        else:
            next_a_index = np.argmax(q_table[next_state])

        # SARSA 更新
        old_value = q_table[state][a_index]
        next_value = q_table[next_state][next_a_index]
        q_table[state][a_index] = old_value + alpha * \
            (reward + gamma * next_value - old_value)

        state = next_state
        a_index = next_a_index
        steps += 1

    # epsilon 衰減
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# 測試策略
print("======= SARSA 測試策略 =======")
state = env.reset()
while env.terminate_states.get(state) == 'goal':
    state = env.reset()

done = False
steps = 0
env.render()
time.sleep(1)

while not done and steps < 25:
    a_index = np.argmax(q_table[state])
    next_state, reward, done, _ = env.step(a_index)
    print(
        f"狀態 {state} + 動作 {actions[a_index]} → 狀態 {next_state}，獎勵: {reward}, 結束: {done}")
    state = next_state
    steps += 1
    env.render()
    time.sleep(0.5)

env.close()
