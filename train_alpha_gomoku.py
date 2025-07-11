"""
AlphaGomokuメインファイル
"""

import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha_gomoku import AlphaZero
from alpha_gomoku.utils import get_args


def main():
    """メイン関数"""
    args = get_args()
    
    # AlphaZeroの初期化
    alpha_zero = AlphaZero(
        board_size=args.board_size,
        num_iterations=args.iterations,
        num_self_play_games=args.games,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        initial_lr=args.lr
    )
    
    # トレーニング開始
    print(f"AlphaGomoku Training Started")
    print(f"Board Size: {args.board_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Games per Iteration: {args.games}")
    print(f"Learning Rate: {args.lr}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"Log Directory: {args.log_dir}")
    
    try:
        alpha_zero.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
