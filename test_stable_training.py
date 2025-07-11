"""
安定化された学習のテスト用スクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_gomoku.training.trainer import AlphaZero
from alpha_gomoku.config import *

def test_stable_training():
    """安定化された学習をテスト"""
    print("=== 安定化された学習のテスト ===")
    
    # 小さなボードサイズでテスト
    trainer = AlphaZero(
        board_size=5,
        num_iterations=5,
        num_self_play_games=8,
        checkpoint_dir='models_stable_test',
        log_dir='logs_stable_test',
        initial_lr=0.0001
    )
    
    print(f"設定情報:")
    print(f"  ボードサイズ: {trainer.board_size}")
    print(f"  イテレーション数: {trainer.num_iterations}")
    print(f"  自己対戦ゲーム数: {trainer.num_self_play_games}")
    print(f"  初期学習率: {trainer.initial_lr}")
    print(f"  バッチサイズ: {TRAINING_BATCH_SIZE}")
    print(f"  エポック数: {TRAINING_EPOCHS}")
    print(f"  リプレイバッファ容量: {REPLAY_BUFFER_CAPACITY}")
    
    # 学習開始
    trainer.train()
    
    print("\n=== 学習完了 ===")
    print(f"最終Policy Loss: {trainer.policy_loss_history[-1]:.6f}")
    print(f"最終Value Loss: {trainer.value_loss_history[-1]:.6f}")
    print(f"安定性チェック: {trainer.loss_stabilizer.is_loss_stable()}")

if __name__ == "__main__":
    test_stable_training()
