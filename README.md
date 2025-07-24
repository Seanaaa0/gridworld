# 🧠 GridWorld: Reinforcement Learning Playground

This project implements a customizable GridWorld environment for training RL agents using tabular reinforcement learning methods like Q-learning and SARSA.

It supports variable grid sizes, deterministic transitions, goal and pit placement, and visual rendering of the agent’s behavior.

---

## 🚀 Features

- ✅ 10x10 and NxN grid environments  
- ✅ Deterministic transition system (up/down/left/right)  
- ✅ Q-learning and SARSA training support  
- ✅ CLI: supports alpha, gamma, epsilon, episodes  
- ✅ Visualization of agent behavior and learned path  
- ✅ Save/load Q-table in `.npy` format  

---

## 📂 Folder Structure

```
grid/
├── envs/                 # GridEnv10x10 / GridEnvNxN classes
├── run/                  # Training and evaluation scripts
├── outputs/              # Q-table files and training results
├── visual/               # (Optional) rendering helper functions
├── requirements.txt
└── .gitignore
```

---

## 💻 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train on 10x10 grid

```bash
python run/run_10x10_cli.py --episodes 3000 --alpha 0.1 --gamma 0.9 --epsilon 0.1 --render
```

---

## 🧠 Use Case

This project serves as:

- An educational sandbox for tabular RL algorithms  
- A training data generator for future GPT-based planning systems  
- A reproducible environment for debugging policies in grid-based navigation tasks

---

## 🔮 Future Work

- Train GPT model to predict best action/goal based on partial trajectories  
- Integrate offline RL dataset and convert to `jsonl` for fine-tuning  
- Add support for non-deterministic transitions and partial observability

---

## 👤 Author

Developed by [@Seanaaa0](https://github.com/Seanaaa0)  
Self-directed AI engineer focused on RL, LLM fine-tuning, and agent design.
