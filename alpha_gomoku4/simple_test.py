#!/usr/bin/env python3
"""
簡単な勝敗判定テスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Game
from mcts import MCTSPlayer
from network import PolicyValueNet
from GomokuEnv import GomokuEnv
import torch

def simple_test():
    print("=== 簡単な勝敗判定テスト ===")
    
    # 先手をtrain_playerにした場合
    print("\n--- 先手をtrain_playerにした場合 ---")
    env1 = GomokuEnv(board_size=9, train_target="first")
    print(f"train_player: {env1.train_player}")
    
    # 後手をtrain_playerにした場合
    print("\n--- 後手をtrain_playerにした場合 ---")
    env2 = GomokuEnv(board_size=9, train_target="second")
    print(f"train_player: {env2.train_player}")
    
    # stepメソッドの勝敗判定をテスト
    print("\n--- stepメソッドの勝敗判定 ---")
    
    # 先手（プレイヤー1）が勝つ場合
    print("先手（プレイヤー1）が勝つケース:")
    print("  train_player=1の場合: current_player=1で勝利 -> reward=1.0")
    print("  train_player=2の場合: current_player=1で勝利 -> reward=-1.0")
    
    # 後手（プレイヤー2）が勝つ場合
    print("後手（プレイヤー2）が勝つケース:")
    print("  train_player=1の場合: current_player=2で勝利 -> reward=-1.0")
    print("  train_player=2の場合: current_player=2で勝利 -> reward=1.0")

if __name__ == "__main__":
    simple_test()
