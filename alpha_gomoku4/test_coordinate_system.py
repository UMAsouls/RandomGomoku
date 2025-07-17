#!/usr/bin/env python3
"""
座標系確認用のテストスクリプト
手動で数手打って座標系を確認する
"""

import numpy as np
import sys
import os

# 現在のディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GomokuEnv import GomokuEnv
from game import Game

def test_coordinate_system():
    """座標系を確認するテスト"""
    print("=== 座標系確認テスト ===")
    
    # 環境とゲームの初期化
    env = GomokuEnv(board_size=8)
    game = Game(env, board_size=8)
    
    # 最初の状態を確認
    print("\n初期状態:")
    env.board.PrintBoard()
    initial_state = game.make_state(env.board.GetBoardInt())
    
    # 手動で石を置いてテスト
    test_moves = [
        (3, 3),  # 中央付近
        (4, 4),  # 中央付近
        (2, 5),  # 別の位置
        (5, 2),  # 別の位置
    ]
    
    for move_num, (x, y) in enumerate(test_moves):
        print(f"\n=== 手番 {move_num + 1}: ({x}, {y}) に石を置く ===")
        print(f"現在のプレイヤー: {env.current_player}")
        
        # 手を打つ
        try:
            _, reward, done, info = env.step((x, y))
            print(f"手を打った後の状態:")
            env.board.PrintBoard()
            
            # 座標系確認のため make_state を呼び出し
            state = game.make_state(env.board.GetBoardInt())
            
            print(f"報酬: {reward}, 終了: {done}")
            if done:
                print(f"勝者: {info.get('win_player', 'なし')}")
                break
                
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            break
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    test_coordinate_system()
