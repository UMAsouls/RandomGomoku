#!/usr/bin/env python3
"""
並列AlphaZeroと従来のAlphaZeroの性能比較ベンチマーク

このスクリプトは、並列版と従来版のAlphaZeroの実行時間を比較します。
"""

import time
import multiprocessing
import numpy as np
from alpha_gomoku_parallel import AlphaZeroParallel
from alphazero import AlphaZero

def benchmark_parallel_training(num_games=10, num_runs=3):
    """並列版AlphaZeroのベンチマーク"""
    print(f"=== 並列版AlphaZeroベンチマーク ===")
    print(f"ゲーム数: {num_games}, 実行回数: {num_runs}")
    
    times = []
    
    for run in range(num_runs):
        print(f"\n--- 実行 {run + 1}/{num_runs} ---")
        
        # 並列AlphaZeroを初期化
        trainer = AlphaZeroParallel()
        trainer.parallel_game_count = num_games
        
        # 実行時間を測定
        start_time = time.time()
        collected_games = trainer.collect_selfplay_data_parallel(num_games)
        end_time = time.time()
        
        execution_time = end_time - start_time
        times.append(execution_time)
        
        print(f"実行時間: {execution_time:.2f}秒")
        print(f"収集したゲーム数: {collected_games}")
        print(f"1ゲームあたりの平均時間: {execution_time/max(collected_games, 1):.2f}秒")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n並列版結果:")
    print(f"  平均実行時間: {avg_time:.2f} ± {std_time:.2f}秒")
    print(f"  最速: {min(times):.2f}秒")
    print(f"  最遅: {max(times):.2f}秒")
    print(f"  使用ワーカー数: {trainer.num_workers}")
    
    return avg_time, std_time

def benchmark_sequential_training(num_games=10, num_runs=3):
    """従来版AlphaZeroのベンチマーク"""
    print(f"\n=== 従来版AlphaZeroベンチマーク ===")
    print(f"ゲーム数: {num_games}, 実行回数: {num_runs}")
    
    times = []
    
    for run in range(num_runs):
        print(f"\n--- 実行 {run + 1}/{num_runs} ---")
        
        # 従来のAlphaZeroを初期化
        trainer = AlphaZero()
        trainer.play_batch_size = 1  # 1ゲームずつ実行
        
        # 実行時間を測定
        start_time = time.time()
        
        # データバッファをクリア
        trainer.date_buffer.clear()
        
        # 指定されたゲーム数だけ実行
        for i in range(num_games):
            trainer.collect_selfplay_data(1)
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        times.append(execution_time)
        
        print(f"実行時間: {execution_time:.2f}秒")
        print(f"収集したゲーム数: {num_games}")
        print(f"1ゲームあたりの平均時間: {execution_time/num_games:.2f}秒")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n従来版結果:")
    print(f"  平均実行時間: {avg_time:.2f} ± {std_time:.2f}秒")
    print(f"  最速: {min(times):.2f}秒")
    print(f"  最遅: {max(times):.2f}秒")
    
    return avg_time, std_time

def main():
    """メイン実行関数"""
    print("AlphaZero並列処理ベンチマーク")
    print("=" * 50)
    
    # システム情報を表示
    cpu_count = multiprocessing.cpu_count()
    print(f"CPUコア数: {cpu_count}")
    
    # マルチプロセシングのスタートメソッドを設定
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # ベンチマーク設定
    num_games = 8  # テスト用にゲーム数を減らす
    num_runs = 2   # テスト用に実行回数を減らす
    
    print(f"\nベンチマーク設定:")
    print(f"  ゲーム数: {num_games}")
    print(f"  実行回数: {num_runs}")
    print(f"  CPU使用可能コア数: {cpu_count}")
    
    try:
        # 並列版のベンチマーク
        parallel_avg, parallel_std = benchmark_parallel_training(num_games, num_runs)
        
        # 従来版のベンチマーク
        sequential_avg, sequential_std = benchmark_sequential_training(num_games, num_runs)
        
        # 結果の比較
        print(f"\n" + "=" * 50)
        print(f"=== 性能比較結果 ===")
        print(f"並列版:   {parallel_avg:.2f} ± {parallel_std:.2f}秒")
        print(f"従来版:   {sequential_avg:.2f} ± {sequential_std:.2f}秒")
        
        if parallel_avg < sequential_avg:
            speedup = sequential_avg / parallel_avg
            print(f"\n並列版が {speedup:.2f}倍 高速です！")
        else:
            slowdown = parallel_avg / sequential_avg
            print(f"\n並列版が {slowdown:.2f}倍 遅くなりました")
        
        # 効率性の計算
        efficiency = (sequential_avg / parallel_avg) / (cpu_count - 1) * 100
        print(f"並列効率: {efficiency:.1f}% (理論値: {100/(cpu_count-1):.1f}%)")
        
    except Exception as e:
        print(f"ベンチマーク実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
