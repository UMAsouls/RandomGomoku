#!/usr/bin/env python3
"""
AlphaZero五目並べの本格学習用スクリプト
修正されたパラメータで効果的な学習を行います
"""

from alpha_gomoku import AlphaZero

def main():
    print("AlphaZero五目並べ - 本格学習")
    
    # 推奨パラメータ（8x8盤面）
    alpha_zero = AlphaZero(
        board_size=8,               # 標準サイズ
        num_iterations=1000,         # 十分なイテレーション数
        num_self_play_games=32*30,     # 適度なゲーム数
        initial_lr=0.001           # 修正された学習率
    )
    
    print("本格学習開始...")
    alpha_zero.train()
    print("学習完了!")
    
    # 人間との対戦テスト
    print("\n人間との対戦を開始しますか？ (y/n): ", end="")
    response = input()
    if response.lower() == 'y':
        alpha_zero.play_against_human()

if __name__ == "__main__":
    main()
