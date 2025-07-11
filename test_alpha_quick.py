#!/usr/bin/env python3
"""
AlphaZero五目並べの簡単なテスト用スクリプト
小さなパラメータで動作確認を行います
"""

from alpha_gomoku import AlphaZero

def main():
    print("AlphaZero五目並べ - 簡単テスト")
    
    # 小さなパラメータでテスト
    alpha_zero = AlphaZero(
        board_size=6,           # 小さなボードサイズ
        num_iterations=5,       # 少ないイテレーション数
        num_self_play_games=4,  # 少ないゲーム数
        initial_lr=0.001        # 修正：learning_rate → initial_lr
    )
    
    print("テスト開始...")
    alpha_zero.train()
    print("テスト完了!")

if __name__ == "__main__":
    main()
