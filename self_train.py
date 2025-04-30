import matplotlib.pyplot as plt
import numpy as np
import GomokuEnv
from pytorch_dqn import MCTSAgent  # DQNAgentの代わりにMCTSAgentをインポート
from collections import deque
import copy

# === Main ===
episodes = 100  # MCTSは計算コストが高いので少なめに設定

# **エージェントの初期化**
train_agent = MCTSAgent(simulations=50)  # シミュレーション回数は調整可能

# **MCTS vs MCTS の対戦**
for episode in range(episodes):
    # 先手後手を交互に設定
    env = GomokuEnv.GomokuEnv(train_target="first" if episode % 2 == 0 else "second")
    state = env.board.copy()
    done = False
    total_reward = 0


    while not done:
        # MCTSエージェントで行動を選択
        action = train_agent.get_action(state, env.current_player)
        
        next_state, reward, done, info = env.step(action)
        
        # プレイヤー交代（元のコードには明示されていなかった）
        # if not done:
        #     env.current_player = 3 - env.current_player
            
        state = next_state
        total_reward += reward

    # ゲーム終了時に結果表示
    print(f"Black: {env.blackStones} White: {env.whiteStones}")
    print(f"episode: {episode}, total reward: {total_reward}")

# MCTSはモデルを保存する必要がないので、save部分は削除


