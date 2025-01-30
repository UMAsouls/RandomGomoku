import matplotlib.pyplot as plt
import numpy as np
import GomokuEnv
import RandomGomoku
from dqn import DQNAgent
from dqn import RandomAgent
import GomokuEnv
from GomokuEnv import Stone
from collections import deque  # 追加

save_path = './dqn_model'

# === Main ===
episodes = 50000000
sync_interval = 20
env = None

# 初期エージェントの設定
train_agent = DQNAgent()
opponent_agent = RandomAgent()

reward_history = []
percentage_history = []
win_history = deque(maxlen=50)  # 直近50試合の結果を保存するキュー
train_target_win = 0
opponent_win = 0
count = 0

# 学習の実行
for episode in range(episodes):
    # 直近50試合の勝率を計算
    if len(win_history) == 50 and sum(win_history) / 50 > 0.75:
        count += 1
        train_agent.save(save_path)
        train_agent = DQNAgent()
        train_agent.load(save_path)
        opponent_agent = DQNAgent()
        opponent_agent.load(save_path)
        win_history.clear()  # リセットして再スタート
        opponent_agent = RandomAgent()


    # 先手後手を交互に設定
    env = GomokuEnv.GomokuEnv(stone=Stone.BLACK if episode % 2 == 0 else Stone.WHITE)

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = train_agent.get_action(state) if env.current_player == env.train_player else opponent_agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        if done:
            is_win = env.current_player != env.train_player  # 学習エージェントが勝ったかどうか
            win_history.append(1 if is_win else 0)  # 勝ちなら1、負けなら0を追加
            train_target_win += is_win
            opponent_win += not is_win

        train_agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        train_agent.sync_qnet()

    reward_history.append(total_reward)
    win_rate = sum(win_history) / len(win_history) if win_history else 0  # 直近50試合の勝率
    percentage_history.append(win_rate)

    if episode % 1 == 0:
        print(f"episode: {episode}, total reward: {total_reward}")
        print(f"train_target: {train_target_win}, opponent win: {opponent_win}, recent win rate: {win_rate:.2f}")

# モデルの保存
train_agent.save(save_path)

# === Plot ===
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()

plt.xlabel('Episode')
plt.ylabel('Winning Percentage (Last 50)')
plt.plot(range(len(percentage_history)), percentage_history)
plt.show()
