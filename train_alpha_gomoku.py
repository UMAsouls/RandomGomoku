"""
AlphaGomokuメインファイル
"""

import sys
import os

# matplotlibバックエンドを非インタラクティブに設定（Tkinterエラーを回避）
import matplotlib
matplotlib.use('Agg')  # GUIを使用しないバックエンド

# マルチプロセッシングの設定
import torch.multiprocessing as mp

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha_gomoku import AlphaZero
from alpha_gomoku.utils import get_args


def main():
    """メイン関数"""
    # マルチプロセッシングの開始方法を設定
    mp.set_start_method('spawn', force=True)
    
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
        import traceback
        traceback.print_exc()
    finally:
        # リソースの適切なクリーンアップ
        try:
            import matplotlib.pyplot as plt
            plt.close('all')  # すべてのMatplotlibの図を閉じる
        except:
            pass
    
    print("Training completed!")


if __name__ == "__main__":
    # マルチプロセッシング環境での実行を確実にする
    mp.freeze_support()
    main()
