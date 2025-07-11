"""
AlphaGomoku 超軽量テスト版
最小限の設定で動作確認用
"""

import sys
import os
import time

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha_gomoku import AlphaZero
from alpha_gomoku.config import device


def main():
    """メイン関数 - 超軽量テスト用設定"""
    
    # 超軽量テスト用の設定
    ultra_light_config = {
        'board_size': 5,           # 最小盤面
        'num_iterations': 2,       # 最小イテレーション
        'num_self_play_games': 2,  # 最小ゲーム数
        'checkpoint_dir': 'models_ultralight',
        'log_dir': 'logs_ultralight',
        'initial_lr': 0.01
    }
    
    print("=" * 70)
    print("AlphaGomoku Ultra Light Test")
    print("=" * 70)
    print(f"使用デバイス: {device}")
    print(f"Board Size: {ultra_light_config['board_size']}x{ultra_light_config['board_size']}")
    print(f"Iterations: {ultra_light_config['num_iterations']}")
    print(f"Games per Iteration: {ultra_light_config['num_self_play_games']}")
    print(f"Learning Rate: {ultra_light_config['initial_lr']}")
    print("-" * 70)
    print("これは動作確認用の超軽量版です。")
    print("実際の性能テストには適していません。")
    print("=" * 70)
    
    try:
        start_time = time.time()
        
        # AlphaZeroの初期化
        print("\n[1/3] AlphaZeroを初期化中...")
        alpha_zero = AlphaZero(
            board_size=ultra_light_config['board_size'],
            num_iterations=ultra_light_config['num_iterations'],
            num_self_play_games=ultra_light_config['num_self_play_games'],
            checkpoint_dir=ultra_light_config['checkpoint_dir'],
            log_dir=ultra_light_config['log_dir'],
            initial_lr=ultra_light_config['initial_lr']
        )
        print("✓ AlphaZero初期化完了")
        
        # トレーニング実行
        print("\n[2/3] トレーニング開始...")
        alpha_zero.train()
        
        training_time = time.time() - start_time
        print(f"\n✓ トレーニング完了 (所要時間: {training_time:.2f}秒)")
        
        # 簡単な動作テスト
        print("\n[3/3] 動作テスト...")
        print("モデルの基本的な動作をテストします。")
        
        # テスト用の盤面を作成
        import numpy as np
        test_state = np.zeros((ultra_light_config['board_size'], ultra_light_config['board_size']))
        
        # MCTSテスト
        from alpha_gomoku.mcts import MCTS
        mcts = MCTS(alpha_zero.model, num_simulations=10)  # 最小シミュレーション
        
        print("MCTSで手を検索中...")
        policy, value = mcts.search(test_state)
        
        print(f"✓ 方策計算完了: {policy.shape}")
        print(f"✓ 価値計算完了: {value:.4f}")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Ultra Light Test 完了!")
        print(f"総所要時間: {total_time:.2f}秒")
        print("=" * 70)
        
        # 対戦テストオプション
        test_play = input("\n簡単な対戦テストを行いますか？ (y/n): ").lower()
        if test_play == 'y':
            print("\n対戦テストを開始します...")
            print("注意: 軽量版のため、AIの強さは限定的です。")
            alpha_zero.play_against_human()
            
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("テスト中断")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"エラー発生: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
    
    print("\nUltra Light Test 終了")


if __name__ == "__main__":
    main()
