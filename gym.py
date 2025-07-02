import gym
from gym.envs.registration import register
from grid_env import GridEnv

# 註冊你的自定義環境
register(
    id='GridWorld-v0',
    entry_point='grid_env:GridEnv',
)

# 使用 render_mode="human" 會觸發你的 render() 函數
env = gym.make('GridWorld-v0')

# 初始化環境
state = env.reset()
done = False

for _ in range(20):  # 最多執行 20 步
    env.render()
    action = env.getAction()
    a = action[int(random.random()*len(action))]  # 隨機選擇一個動作
    next_state, reward, done, _ = env._step(a)
    print(f"狀態 {state} + 動作 {a} → 狀態 {next_state}，獎勵: {reward}, 結束: {done}")
    state = next_state
    if done:
        break

env.close()
