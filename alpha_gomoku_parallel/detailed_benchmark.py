"""
詳細なパフォーマンス分析とベンチマーク
"""
import time
import multiprocessing
import torch
import sys
import os
import psutil
import threading
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt

# パスを追加
sys.path.append('../alpha_gomoku2')

from alphazero_parallel import AlphaZeroParallel
from alphazero_sequential import AlphaZeroSequential
from parallel_utils import ResourceManager, PerformanceMonitor

class DetailedBenchmark:
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.performance_monitor = PerformanceMonitor()
        
    def benchmark_scalability(self):
        """スケーラビリティのベンチマーク"""
        print("スケーラビリティテストを開始...")
        
        # 異なる並列度でのテスト
        parallel_levels = [1, 2, 4, 8]
        results = {}
        
        for parallel_games in parallel_levels:
            print(f"\n並列度 {parallel_games} でのテスト...")
            
            alpha_zero = AlphaZeroParallel()
            alpha_zero.parallel_games = parallel_games
            
            start_time = time.time()
            
            # 5ゲームの実行時間を測定
            alpha_zero.collect_parallel_selfplay_data(5)
            
            total_time = time.time() - start_time
            results[parallel_games] = total_time
            
            print(f"並列度 {parallel_games}: {total_time:.2f}秒")
        
        return results
    
    def benchmark_memory_usage(self):
        """メモリ使用量のベンチマーク"""
        print("メモリ使用量テストを開始...")
        
        # 逐次処理版のメモリ使用量
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        sequential_alpha = AlphaZeroSequential()
        sequential_alpha.collect_selfplay_data(3)
        
        sequential_memory = process.memory_info().rss / 1024 / 1024  # MB
        sequential_usage = sequential_memory - initial_memory
        
        print(f"逐次処理版メモリ使用量: {sequential_usage:.1f} MB")
        
        # 並列処理版のメモリ使用量
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        parallel_alpha = AlphaZeroParallel()
        parallel_alpha.collect_parallel_selfplay_data(3)
        
        parallel_memory = process.memory_info().rss / 1024 / 1024  # MB
        parallel_usage = parallel_memory - initial_memory
        
        print(f"並列処理版メモリ使用量: {parallel_usage:.1f} MB")
        
        return {
            'sequential': sequential_usage,
            'parallel': parallel_usage
        }
    
    def benchmark_cpu_utilization(self):
        """CPU使用率のベンチマーク"""
        print("CPU使用率テストを開始...")
        
        # CPU使用率を監視するスレッド
        cpu_usage = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_usage.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 並列処理版のテスト
        alpha_zero = AlphaZeroParallel()
        start_time = time.time()
        alpha_zero.collect_parallel_selfplay_data(5)
        end_time = time.time()
        
        monitoring = False
        monitor_thread.join()
        
        avg_cpu_usage = np.mean(cpu_usage)
        max_cpu_usage = np.max(cpu_usage)
        
        print(f"平均CPU使用率: {avg_cpu_usage:.1f}%")
        print(f"最大CPU使用率: {max_cpu_usage:.1f}%")
        print(f"実行時間: {end_time - start_time:.2f}秒")
        
        return {
            'avg_cpu': avg_cpu_usage,
            'max_cpu': max_cpu_usage,
            'execution_time': end_time - start_time
        }
    
    def create_performance_report(self):
        """総合的なパフォーマンスレポートを作成"""
        print("=" * 60)
        print("詳細パフォーマンス分析レポート")
        print("=" * 60)
        
        # システム情報
        print("\n【システム情報】")
        print(f"CPU: {multiprocessing.cpu_count()} cores")
        print(f"GPU: {'利用可能' if torch.cuda.is_available() else '利用不可'}")
        if torch.cuda.is_available():
            print(f"GPU名: {torch.cuda.get_device_name(0)}")
            print(f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        memory = psutil.virtual_memory()
        print(f"システムメモリ: {memory.total / (1024**3):.1f} GB")
        
        # スケーラビリティテスト
        print("\n【スケーラビリティテスト】")
        scalability_results = self.benchmark_scalability()
        
        # 基準時間（並列度1）に対する効率
        base_time = scalability_results[1]
        print(f"並列度別の性能:")
        for parallel_games, time_taken in scalability_results.items():
            efficiency = (base_time / time_taken) / parallel_games * 100
            speedup = base_time / time_taken
            print(f"  並列度 {parallel_games}: {time_taken:.2f}秒 (効率: {efficiency:.1f}%, スピードアップ: {speedup:.2f}x)")
        
        # メモリ使用量テスト
        print("\n【メモリ使用量テスト】")
        memory_results = self.benchmark_memory_usage()
        memory_ratio = memory_results['parallel'] / memory_results['sequential']
        print(f"メモリ使用量比較:")
        print(f"  逐次処理版: {memory_results['sequential']:.1f} MB")
        print(f"  並列処理版: {memory_results['parallel']:.1f} MB")
        print(f"  並列版のメモリ使用量: {memory_ratio:.1f}x")
        
        # CPU使用率テスト
        print("\n【CPU使用率テスト】")
        cpu_results = self.benchmark_cpu_utilization()
        print(f"CPU使用率分析:")
        print(f"  平均CPU使用率: {cpu_results['avg_cpu']:.1f}%")
        print(f"  最大CPU使用率: {cpu_results['max_cpu']:.1f}%")
        print(f"  実行時間: {cpu_results['execution_time']:.2f}秒")
        
        # 推奨事項
        print("\n【推奨事項】")
        optimal_params = self.resource_manager.get_optimal_params()
        print(f"最適な並列設定:")
        print(f"  並列ゲーム数: {optimal_params['game_workers']}")
        print(f"  データ処理ワーカー数: {optimal_params['data_workers']}")
        print(f"  学習バッチサイズ: {optimal_params['training_batch_size']}")
        print(f"  GPU使用: {'推奨' if optimal_params['use_gpu'] else '不要'}")
        
        # 期待される性能向上
        expected_speedup = min(multiprocessing.cpu_count() - 1, 8) * 0.8  # 80%の効率を仮定
        print(f"\n期待される性能向上: {expected_speedup:.1f}x")
        
        print("\n" + "=" * 60)
        print("分析完了")
        print("=" * 60)
        
        return {
            'scalability': scalability_results,
            'memory': memory_results,
            'cpu': cpu_results,
            'optimal_params': optimal_params
        }

def main():
    """メイン実行関数"""
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    benchmark = DetailedBenchmark()
    results = benchmark.create_performance_report()
    
    print("\n詳細分析が完了しました。")
    print("この結果を参考に、最適な並列処理設定を調整してください。")

if __name__ == "__main__":
    main()
