"""
並列処理版と従来版のパフォーマンス比較ベンチマーク
"""
import time
import multiprocessing
import torch
import sys
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# パスを追加
sys.path.append('../alpha_gomoku2')

from alphazero_parallel import AlphaZeroParallel
from parallel_utils import ResourceManager, PerformanceMonitor

def benchmark_parallel_version():
    """並列処理版のベンチマーク"""
    print("並列処理版のベンチマークを開始...")
    
    # 短いトレーニングサイクルでテスト
    alpha_zero = AlphaZeroParallel()
    alpha_zero.game_batch_num = 10  # 短いサイクル
    alpha_zero.check_freq = 5       # 短い評価間隔
    
    start_time = time.time()
    
    # 自己対戦データ収集のテスト（8ゲームを並列実行）
    for i in range(5):
        batch_start = time.time()
        alpha_zero.collect_parallel_selfplay_data(8)  # 8ゲーム並列実行
        batch_time = time.time() - batch_start
        print(f"並列バッチ {i+1}: {batch_time:.2f}秒")
    
    total_time = time.time() - start_time
    print(f"並列処理版総時間: {total_time:.2f}秒")
    
    return total_time

def benchmark_sequential_version():
    """従来版（逐次処理）のベンチマーク"""
    print("従来版（逐次処理）のベンチマークを開始...")
    
    # 真の逐次処理版を使用
    from alphazero_sequential import AlphaZeroSequential
    
    alpha_zero = AlphaZeroSequential()
    
    start_time = time.time()
    
    # 逐次処理でのデータ収集テスト（8ゲームを逐次実行）
    for i in range(5):
        batch_start = time.time()
        alpha_zero.collect_selfplay_data(8)  # 8ゲームを逐次実行
        batch_time = time.time() - batch_start
        print(f"逐次バッチ {i+1}: {batch_time:.2f}秒")
    
    total_time = time.time() - start_time
    print(f"逐次処理版総時間: {total_time:.2f}秒")
    
    return total_time
    
    total_time = time.time() - start_time
    print(f"逐次処理版総時間: {total_time:.2f}秒")
    
    return total_time

def compare_performance():
    """パフォーマンスを比較"""
    print("=" * 60)
    print("パフォーマンス比較ベンチマーク")
    print("=" * 60)
    
    # システム情報を表示
    resource_manager = ResourceManager()
    
    # 各バージョンのベンチマーク実行
    try:
        parallel_time = benchmark_parallel_version()
        print("\n" + "-" * 40)
        sequential_time = benchmark_sequential_version()
        
        print("\n" + "=" * 60)
        print("ベンチマーク結果:")
        print(f"  並列処理版: {parallel_time:.2f}秒")
        print(f"  逐次処理版: {sequential_time:.2f}秒")
        
        if sequential_time > 0:
            speedup = sequential_time / parallel_time
            print(f"  スピードアップ: {speedup:.2f}x")
            print(f"  時間短縮: {((sequential_time - parallel_time) / sequential_time * 100):.1f}%")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"ベンチマーク実行中にエラーが発生しました: {e}")

def analyze_resource_usage():
    """リソース使用量の分析"""
    print("\nリソース使用量の分析:")
    
    # CPU使用率
    cpu_count = multiprocessing.cpu_count()
    print(f"  利用可能CPU数: {cpu_count}")
    
    # GPU情報
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  利用可能GPU数: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_properties = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {gpu_properties.name}")
            print(f"    メモリ: {gpu_properties.total_memory / (1024**3):.1f} GB")
    else:
        print("  GPU: 利用不可")
    
    # メモリ情報
    import psutil
    memory = psutil.virtual_memory()
    print(f"  システムメモリ: {memory.total / (1024**3):.1f} GB")
    print(f"  利用可能メモリ: {memory.available / (1024**3):.1f} GB")

def main():
    """メイン実行関数"""
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    print("並列処理版AlphaZero ベンチマークスイート")
    print("=" * 60)
    
    # システムリソースの分析
    analyze_resource_usage()
    
    # パフォーマンス比較
    compare_performance()
    
    print("\nベンチマークが完了しました。")

if __name__ == "__main__":
    main()
