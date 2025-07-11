"""
AlphaGomokuと人間の対戦
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
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    print(f"AlphaGomoku vs Human")
    print(f"Board Size: {args.board_size}")
    
    try:
        alpha_zero.play_against_human()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Game error: {e}")
    
    print("Game completed!")


if __name__ == "__main__":
    main()
