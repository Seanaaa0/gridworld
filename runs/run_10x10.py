from datetime import datetime
import os
import gym
from gym.envs.registration import register
from grid_10x10 import GridEnv10x10
import numpy as np
import random
import time

# 註冊環境
register(
    id='GridWorld10x10-v0',
    entry_point='grid_10x10:GridEnv10x10',
)

# 建立環境
env = gym.make('GridWorld10x10-v0')
actions = env.getAction()
num_states = len(env.getStates())
num_actions = len(actions)

q_table = np.zeros((num_states + 1, num_actions))

# SARSA 參數
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
episodes = 5000

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

    while not done and steps < 100:
        next_state, reward, done, _ = env.step(a_index)

        if random.uniform(0, 1) < epsilon:
            next_a_index = random.randint(0, num_actions - 1)
        else:
            next_a_index = np.argmax(q_table[next_state])

        # SARSA 更新
        old_value = q_table[state][a_index]
        next_value = q_table[next_state][next_a_index]
        q_table[state][a_index] = old_value + alpha * (
            reward + gamma * next_value - old_value
        )

        state = next_state
        a_index = next_a_index
        steps += 1

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# 測試 SARSA 策略（有探索性）
print("======= SARSA 測試策略（含 test_epsilon 探索）=======")
state = env.reset()
while env.terminate_states.get(state) == 'goal':
    state = env.reset()

done = False
steps = 0
test_epsilon = 0.65  # 測試階段仍然有 30% 的隨機探索機會
env.render()
time.sleep(1)

while not done and steps < 50:
    if random.uniform(0, 1) < test_epsilon:
        a_index = random.randint(0, num_actions - 1)
    else:
        a_index = np.argmax(q_table[state])

    next_state, reward, done, _ = env.step(a_index)
    print(
        f"狀態 {state} + 動作 {actions[a_index]} → 狀態 {next_state}，獎勵: {reward}, 結束: {done}")
    state = next_state
    steps += 1
    env.render()
    time.sleep(0.5)

env.close()


# 檔案路徑
save_dir = "C:/Users/seana/10x10"
os.makedirs(save_dir, exist_ok=True)

# 產生時間戳記
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"q_table_{timestamp}.npy"
q_table_path = os.path.join(save_dir, filename)

# 儲存 Q-table
np.save(q_table_path, q_table)
print(f"Q-table 已儲存到：{q_table_path}")
