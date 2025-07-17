#!/usr/bin/env python3
"""
先手後手の勝敗判定テストスクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Game
from mcts import MCTSPlayer
from network import PolicyValueNet
from GomokuEnv import GomokuEnv
import torch

def test_win_logic():
    """
    勝敗判定のロジックをテストします
    """
    print("=== 勝敗判定テスト開始 ===")
    
    # 小さいボードサイズでテスト
    board_size = 9
    
    # 先手をtrain_playerにした場合
    print("\n--- 先手をtrain_playerにした場合 ---")
    env_first = GomokuEnv(board_size=board_size, train_target="first")
    game_first = Game(env_first, board_size=board_size)
    
    # 軽量なネットワークを作成
    policy_value_net = PolicyValueNet(board_size, env=env_first)
    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                            c_puct=1.0,
                            n_playout=20,  # 少ないプレイアウト数でテスト
                            is_selfplay=True)
    
    # 数回のゲームを実行
    first_wins = 0
    second_wins = 0
    draws = 0
    
    for i in range(3):
        print(f"\n--- ゲーム {i+1} (先手=train_player) ---")
        winners, play_data = game_first.start_self_play(mcts_player, is_shown=True, temp=0.1)
        
        if len(winners) > 0:
            last_winner = winners[-1]
            if last_winner > 0:
                first_wins += 1
                print(f"結果: 先手(train_player)勝利")
            elif last_winner < 0:
                second_wins += 1
                print(f"結果: 後手勝利")
            else:
                draws += 1
                print(f"結果: 引き分け")
        
        mcts_player.reset_player()
    
    print(f"\n先手=train_player の結果: 先手勝利={first_wins}, 後手勝利={second_wins}, 引き分け={draws}")
    
    # 後手をtrain_playerにした場合
    print("\n--- 後手をtrain_playerにした場合 ---")
    env_second = GomokuEnv(board_size=board_size, train_target="second")
    game_second = Game(env_second, board_size=board_size)
    
    policy_value_net2 = PolicyValueNet(board_size, env=env_second)
    mcts_player2 = MCTSPlayer(policy_value_net2.policy_value_fn,
                             c_puct=1.0,
                             n_playout=20,
                             is_selfplay=True)
    
    first_wins2 = 0
    second_wins2 = 0
    draws2 = 0
    
    for i in range(3):
        print(f"\n--- ゲーム {i+1} (後手=train_player) ---")
        winners, play_data = game_second.start_self_play(mcts_player2, is_shown=True, temp=0.1)
        
        if len(winners) > 0:
            last_winner = winners[-1]
            if last_winner > 0:
                second_wins2 += 1
                print(f"結果: 後手(train_player)勝利")
            elif last_winner < 0:
                first_wins2 += 1
                print(f"結果: 先手勝利")
            else:
                draws2 += 1
                print(f"結果: 引き分け")
        
        mcts_player2.reset_player()
    
    print(f"\n後手=train_player の結果: 先手勝利={first_wins2}, 後手勝利={second_wins2}, 引き分け={draws2}")
    
    print("\n=== 勝敗判定テスト完了 ===")

if __name__ == "__main__":
    test_win_logic()
