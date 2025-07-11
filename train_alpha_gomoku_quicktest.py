"""
AlphaGomoku クイックテスト版
短時間でのテストと動作確認用
"""

import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha_gomoku import AlphaZero
from alpha_gomoku.utils import get_args


def main():
    """メイン関数 - クイックテスト用設定"""
    args = get_args()
    
    # クイックテスト用の設定上書き
    quick_config = {
        'board_size': 6,           # 小さな盤面でテスト
        'num_iterations': 3,       # 少ないイテレーション
        'num_self_play_games': 4,  # 少ないゲーム数
        'checkpoint_dir': 'models_test',  # テスト用ディレクトリ
        'log_dir': 'logs_test',    # テスト用ログディレクトリ
        'initial_lr': 0.01         # 高めの学習率で早期収束
    }
    
    # AlphaZeroの初期化（クイックテスト設定）
    alpha_zero = AlphaZero(
        board_size=quick_config['board_size'],
        num_iterations=quick_config['num_iterations'],
        num_self_play_games=quick_config['num_self_play_games'],
        checkpoint_dir=quick_config['checkpoint_dir'],
        log_dir=quick_config['log_dir'],
        initial_lr=quick_config['initial_lr']
    )
    
    # トレーニング開始
    print("=" * 60)
    print("AlphaGomoku QuickTest Training Started")
    print("=" * 60)
    print(f"Board Size: {quick_config['board_size']} (小さな盤面)")
    print(f"Iterations: {quick_config['num_iterations']} (少ないイテレーション)")
    print(f"Games per Iteration: {quick_config['num_self_play_games']} (少ないゲーム数)")
    print(f"Learning Rate: {quick_config['initial_lr']} (高めの学習率)")
    print(f"Checkpoint Directory: {quick_config['checkpoint_dir']}")
    print(f"Log Directory: {quick_config['log_dir']}")
    print("-" * 60)
    print("注意: これはクイックテスト版です。実際の訓練には適していません。")
    print("=" * 60)
    
    try:
        alpha_zero.train()
        print("\n" + "=" * 60)
        print("QuickTest Training Completed Successfully!")
        print("=" * 60)
        
        # 簡単な対戦テスト
        test_human_play = input("\n人間との対戦テストを行いますか？ (y/n): ").lower()
        if test_human_play == 'y':
            print("\n人間との対戦テストを開始します...")
            alpha_zero.play_against_human()
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Training error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
    
    print("\nQuickTest completed!")


if __name__ == "__main__":
    main()
