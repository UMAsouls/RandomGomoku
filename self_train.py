import matplotlib.pyplot as plt
import numpy as np
import GomokuEnv
import RandomGomoku
from dqn import DQNAgent
import GomokuEnv
from GomokuEnv import Stone
from collections import deque
from agent import RandomAgent
from agent import RuleBasedAgent  
from agent import MinimaxAgent
import copy  

save_path = './dqn_model'  # 最良モデルの保存場所

# === Main ===
episodes = 10000
sync_interval = 20
env = None

# **勝率の閾値**
win_rate_threshold_random = 0.8  # ランダム相手で80%勝てたらDQNと対戦
win_rate_threshold_dqn = 0.95  # DQN同士の対戦で95%勝てたら学習終了
win_rate_lower_bound = 0.1  # 勝率が10%を切ったら復元

# **エージェントの初期化**
train_agent = DQNAgent()
opponent_agent = RandomAgent()  # 最初はランダムエージェント

reward_history = []
percentage_history = []
win_history_maxlen = 50
win_history = deque(maxlen=win_history_maxlen)
train_target_win = 0
opponent_win = 0

is_ai_vs_ai = False  # AI同士の対戦に移行したかどうか
best_win_rate = 0  # 最良の勝率

changed_episode = 0  # ランダム相手からDQN相手に移行したエピソード

# 学習の実行
for episode in range(episodes):

    # **ランダム相手で80%勝ったら AI vs AI に移行**
    if not is_ai_vs_ai:
        win_rate = sum(win_history) / len(win_history) if len(win_history) > 0 else 0
        if win_rate > win_rate_threshold_random and episode > 30:
            print("----------------------------")
            print("AI has learned well against RandomAgent!")
            print("Switching opponent to DQN self-play.")
            print("----------------------------")

            opponent_agent = copy.deepcopy(train_agent)  # 過去のDQNを固定
            is_ai_vs_ai = True  # AI vs AI に移行

    # 先手後手を交互に設定
    env = GomokuEnv.GomokuEnv(train_target="first" if episode % 2 == 0 else "second")

    state = env.board.copy()
    done = False
    total_reward = 0

    while not done:
        # AI vs AI で戦う場合
        action = train_agent.get_action(state, env.current_player) if env.current_player == env.train_player else opponent_agent.get_action(state, env.current_player)

        next_state, reward, done, info = env.step(action)

        # 次のプレイヤーに交代
        env.current_player = 3 - env.current_player
        
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
    win_rate = sum(win_history) / len(win_history) if len(win_history) > 0 else 0  # 直近50試合の勝率

    # **最良の勝率を記録し、保存**
    if win_rate > best_win_rate:
        best_win_rate = win_rate
        print(f"New best win rate: {best_win_rate:.2f}, saving model.")
        train_agent.save(save_path)

    # **勝率が10%を下回ったら復元**
    if is_ai_vs_ai and win_rate < win_rate_lower_bound and episode - changed_episode > 50:
        print("----------------------------")
        print(f"Win rate dropped below {win_rate_lower_bound * 100:.0f}%, restoring best model.")
        print("----------------------------")
        win_history.clear()
        changed_episode = episode
        train_agent.load(save_path)  # 過去の最良モデルを復元

    # **AI vs AI の勝率が95%を超えたらモデルを保存して終了**
    if is_ai_vs_ai and win_rate > win_rate_threshold_dqn and episode > 50:
        print("----------------------------")
        print("DQN has achieved high win rate against itself!")
        print("Saving model and stopping training.")
        print("----------------------------")
        win_history.clear()
        train_agent.save(save_path)
        break

    percentage_history.append(win_rate)

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
