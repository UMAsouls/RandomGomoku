"""
AlphaGomoku テストスイート
様々なテストレベルを選択可能
"""

import sys
import os
import time
import argparse

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha_gomoku import AlphaZero
from alpha_gomoku.config import device


def get_test_args():
    """テスト用コマンドライン引数を取得"""
    parser = argparse.ArgumentParser(description='AlphaGomoku Test Suite')
    parser.add_argument('--level', type=str, default='quick', 
                       choices=['ultra', 'quick', 'normal'],
                       help='Test level: ultra(最軽量), quick(軽量), normal(通常)')
    parser.add_argument('--play', action='store_true',
                       help='トレーニング後に対戦テストを実行')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細なログを出力')
    return parser.parse_args()


def get_config(level):
    """テストレベルに応じた設定を取得"""
    configs = {
        'ultra': {
            'board_size': 5,
            'num_iterations': 1,
            'num_self_play_games': 2,
            'checkpoint_dir': 'models_ultra',
            'log_dir': 'logs_ultra',
            'initial_lr': 0.01,
            'description': '超軽量版 - 最小限の動作確認'
        },
        'quick': {
            'board_size': 6,
            'num_iterations': 3,
            'num_self_play_games': 4,
            'checkpoint_dir': 'models_quick',
            'log_dir': 'logs_quick',
            'initial_lr': 0.01,
            'description': '軽量版 - 基本的な動作確認'
        },
        'normal': {
            'board_size': 8,
            'num_iterations': 5,
            'num_self_play_games': 8,
            'checkpoint_dir': 'models_normal',
            'log_dir': 'logs_normal',
            'initial_lr': 0.005,
            'description': '通常版 - 標準的なテスト'
        }
    }
    return configs[level]


def main():
    """メイン関数"""
    args = get_test_args()
    config = get_config(args.level)
    
    print("=" * 80)
    print(f"AlphaGomoku Test Suite - {args.level.upper()} Level")
    print("=" * 80)
    print(f"使用デバイス: {device}")
    print(f"テストレベル: {config['description']}")
    print(f"Board Size: {config['board_size']}x{config['board_size']}")
    print(f"Iterations: {config['num_iterations']}")
    print(f"Games per Iteration: {config['num_self_play_games']}")
    print(f"Learning Rate: {config['initial_lr']}")
    print(f"Checkpoint Directory: {config['checkpoint_dir']}")
    print(f"Log Directory: {config['log_dir']}")
    print("-" * 80)
    
    if args.level == 'ultra':
        print("⚠️  注意: 超軽量版は動作確認のみを目的としています。")
    elif args.level == 'quick':
        print("ℹ️  情報: 軽量版は基本的な動作確認を目的としています。")
    else:
        print("ℹ️  情報: 通常版は標準的なテストを実行します。")
    
    print("=" * 80)
    
    try:
        start_time = time.time()
        
        # AlphaZeroの初期化
        print("\n📚 AlphaZeroを初期化中...")
        alpha_zero = AlphaZero(
            board_size=config['board_size'],
            num_iterations=config['num_iterations'],
            num_self_play_games=config['num_self_play_games'],
            checkpoint_dir=config['checkpoint_dir'],
            log_dir=config['log_dir'],
            initial_lr=config['initial_lr']
        )
        print("✅ AlphaZero初期化完了")
        
        # トレーニング実行
        print("\n🚀 トレーニング開始...")
        alpha_zero.train()
        
        training_time = time.time() - start_time
        print(f"\n✅ トレーニング完了 (所要時間: {training_time:.2f}秒)")
        
        # 基本動作テスト
        print("\n🔍 基本動作テスト...")
        
        # テスト用の盤面を作成
        import numpy as np
        test_state = np.zeros((config['board_size'], config['board_size']))
        
        # MCTSテスト
        from alpha_gomoku.mcts import MCTS
        sim_count = 10 if args.level == 'ultra' else 50 if args.level == 'quick' else 100
        mcts = MCTS(alpha_zero.model, num_simulations=sim_count)
        
        if args.verbose:
            print(f"MCTSで手を検索中... (シミュレーション数: {sim_count})")
        
        policy, value = mcts.search(test_state)
        
        print(f"✅ 方策計算完了: {policy.shape}")
        print(f"✅ 価値計算完了: {value:.4f}")
        
        # 統計情報
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"🎉 Test Suite 完了!")
        print(f"📊 統計情報:")
        print(f"   - 総所要時間: {total_time:.2f}秒")
        print(f"   - 平均イテレーション時間: {training_time/config['num_iterations']:.2f}秒")
        print(f"   - 使用メモリ: {config['board_size']**2 * config['num_self_play_games']} positions")
        print("=" * 80)
        
        # 対戦テスト
        if args.play:
            print("\n🎮 対戦テストを開始します...")
            if args.level == 'ultra':
                print("⚠️  注意: 超軽量版のため、AIの強さは非常に限定的です。")
            alpha_zero.play_against_human()
        else:
            test_play = input("\n対戦テストを行いますか？ (y/n): ").lower()
            if test_play == 'y':
                print("\n🎮 対戦テストを開始します...")
                if args.level == 'ultra':
                    print("⚠️  注意: 超軽量版のため、AIの強さは非常に限定的です。")
                alpha_zero.play_against_human()
            
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("❌ テスト中断")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"💥 エラー発生: {e}")
        print("=" * 80)
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    print("\n👋 Test Suite 終了")


if __name__ == "__main__":
    main()
