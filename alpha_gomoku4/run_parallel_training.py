#!/usr/bin/env python3
"""
並列AlphaZeroトレーニングの実行スクリプト

このスクリプトは、CPUコア数に応じて自己対戦を並列実行するAlphaZeroトレーニングを開始します。
"""

import sys
import os
import multiprocessing
import argparse
from collections import deque
from alpha_gomoku_parallel import AlphaZeroParallel

def main():
    parser = argparse.ArgumentParser(description='並列AlphaZeroトレーニング')
    parser.add_argument('--workers', type=int, default=None,
                        help='並列ワーカー数（デフォルト: CPU数-1）')
    parser.add_argument('--games-per-batch', type=int, default=8,
                        help='バッチあたりのゲーム数（デフォルト: 4）')
    parser.add_argument('--total-batches', type=int, default=5000,
                        help='総バッチ数（デフォルト: 5000）')
    parser.add_argument('--playout', type=int, default=400,
                        help='MCTSプレイアウト数（デフォルト: 400）')
    parser.add_argument('--check-freq', type=int, default=50,
                        help='モデル評価の頻度（デフォルト: 100）')
    parser.add_argument('--buffer-size', type=int, default=None,
                        help='経験再生バッファサイズ（デフォルト: 自動調整）')
    
    args = parser.parse_args()
    
    # マルチプロセシングのスタートメソッドを設定
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 既に設定されている場合は無視
    
    # CPU数を表示
    cpu_count = multiprocessing.cpu_count()
    print(f"検出されたCPUコア数: {cpu_count}")
    
    # 並列AlphaZeroトレーニングを初期化
    trainer = AlphaZeroParallel()
    
    # コマンドライン引数に基づいてパラメータを設定
    if args.workers is not None:
        trainer.num_workers = min(args.workers, cpu_count - 1)
    
    trainer.parallel_game_count = args.games_per_batch
    trainer.game_batch_num = args.total_batches
    trainer.n_playout = args.playout
    trainer.check_freq = args.check_freq
    
    # バッファサイズの設定
    if args.buffer_size is not None:
        trainer.buffer_size = args.buffer_size
        trainer.date_buffer = deque(maxlen=trainer.buffer_size)
        print(f"バッファサイズを手動設定: {trainer.buffer_size}")
    else:
        # 自動調整（既に__init__で実行済み）
        pass
    
    print(f"トレーニング設定:")
    print(f"  並列ワーカー数: {trainer.num_workers}")
    print(f"  バッチあたりのゲーム数: {trainer.parallel_game_count}")
    print(f"  総バッチ数: {trainer.game_batch_num}")
    print(f"  MCTSプレイアウト数: {trainer.n_playout}")
    print(f"  モデル評価頻度: {trainer.check_freq}")
    print(f"  経験再生バッファサイズ: {trainer.buffer_size}")
    
    # トレーニングを開始
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nトレーニングが中断されました。")
    except Exception as e:
        print(f"トレーニング中にエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
