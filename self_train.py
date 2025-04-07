import matplotlib.pyplot as plt
import numpy as np
import GomokuEnv
from pytorch_dqn import DQNAgent
from collections import deque
import copy

save_path = './dqn_model'  # 最良モデルの保存場所

# === Main ===
episodes = 10000
sync_interval = 20


# **エージェントの初期化**
train_agent = DQNAgent()
opponent_agent = train_agent  # AI同士の対戦、同じQ-networkを使用


# **AI vs AI の学習開始**
for episode in range(episodes):

    # 先手後手を交互に設定
    env = GomokuEnv.GomokuEnv(train_target="first" if episode % 2 == 0 else "second")

    state = env.board.copy()
    done = False
    total_reward = 0

    while not done:
        # **両者とも同じ Q-network を使う**
        action = train_agent.get_action(state, env.current_player)

        next_state, reward, done, info = env.step(action)

        train_agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        train_agent.sync_qnet()
        
    print(f"episode: {episode}, total reward: {total_reward}")

# モデルの保存
train_agent.save(save_path)


