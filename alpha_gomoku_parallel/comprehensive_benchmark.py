"""
AlphaZero実装間の包括的ベンチマーク比較ツール
alpha_gomoku2 (元の実装) vs alpha_gomoku_parallel (並列処理版)
"""
import os
import sys
import time
import multiprocessing
import torch
import numpy as np
import matplotlib.pyplot as plt
import psutil
import threading
from concurrent.futures import ProcessPoolExecutor
import json
from datetime import datetime
import traceback

# パスを追加
sys.path.append('../alpha_gomoku2')
sys.path.append('../alpha_gomoku_parallel')

class ComprehensiveBenchmark:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
        # システム情報を取得
        self.system_info = self._get_system_info()
        
        print("=" * 70)
        print("AlphaZero実装間の包括的ベンチマーク比較")
        print("=" * 70)
        print(f"開始時刻: {self.start_time}")
        print(f"システム情報: {self.system_info}")
        print("=" * 70)
    
    def _get_system_info(self):
        """システム情報を取得"""
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0,
            'python_version': sys.version.split()[0],
            'pytorch_version': torch.__version__
        }
    
    def benchmark_original_implementation(self, num_games=5, num_batches=3):
        """alpha_gomoku2の元の実装をベンチマーク"""
        print("\n" + "="*50)
        print("元の実装 (alpha_gomoku2) のベンチマーク")
        print("="*50)
        
        try:
            # 元の実装をインポート
            from alphazero import AlphaZero
            
            alpha_zero = AlphaZero()
            
            # 実行時間を測定
            times = []
            memory_usage = []
            
            for i in range(num_batches):
                print(f"\nバッチ {i+1}/{num_batches} を実行中...")
                
                # メモリ使用量を監視
                process = psutil.Process()
                initial_memory = process.memory_info().rss / (1024**2)  # MB
                
                start_time = time.time()
                
                # 自己対戦データを収集
                alpha_zero.collect_selfplay_data(num_games)
                
                end_time = time.time()
                
                final_memory = process.memory_info().rss / (1024**2)  # MB
                memory_used = final_memory - initial_memory
                
                batch_time = end_time - start_time
                times.append(batch_time)
                memory_usage.append(memory_used)
                
                print(f"バッチ {i+1} 完了: {batch_time:.2f}秒, メモリ使用量: {memory_used:.1f}MB")
            
            # 統計情報を計算
            results = {
                'implementation': 'original',
                'num_games': num_games,
                'num_batches': num_batches,
                'times': times,
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'total_time': sum(times),
                'memory_usage': memory_usage,
                'avg_memory': np.mean(memory_usage),
                'games_per_second': num_games / np.mean(times)
            }
            
            print(f"\n元の実装の結果:")
            print(f"  平均実行時間: {results['avg_time']:.2f}秒")
            print(f"  標準偏差: {results['std_time']:.2f}秒")
            print(f"  総実行時間: {results['total_time']:.2f}秒")
            print(f"  平均メモリ使用量: {results['avg_memory']:.1f}MB")
            print(f"  ゲーム/秒: {results['games_per_second']:.2f}")
            
            return results
            
        except Exception as e:
            print(f"元の実装のベンチマークでエラー: {e}")
            traceback.print_exc()
            return None
    
    def benchmark_parallel_implementation(self, num_games=5, num_batches=3):
        """alpha_gomoku_parallelの並列実装をベンチマーク"""
        print("\n" + "="*50)
        print("並列実装 (alpha_gomoku_parallel) のベンチマーク")
        print("="*50)
        
        try:
            # 並列実装をインポート
            from alphazero_parallel import AlphaZeroParallel
            
            alpha_zero = AlphaZeroParallel()
            
            # 実行時間を測定
            times = []
            memory_usage = []
            cpu_usage = []
            
            for i in range(num_batches):
                print(f"\nバッチ {i+1}/{num_batches} を実行中...")
                
                # CPU使用率を監視
                cpu_monitor = []
                monitoring = True
                
                def monitor_cpu():
                    while monitoring:
                        cpu_monitor.append(psutil.cpu_percent(interval=0.1))
                        time.sleep(0.1)
                
                monitor_thread = threading.Thread(target=monitor_cpu)
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # メモリ使用量を監視
                process = psutil.Process()
                initial_memory = process.memory_info().rss / (1024**2)  # MB
                
                start_time = time.time()
                
                # 並列自己対戦データを収集
                alpha_zero.collect_parallel_selfplay_data(num_games)
                
                end_time = time.time()
                
                monitoring = False
                monitor_thread.join()
                
                final_memory = process.memory_info().rss / (1024**2)  # MB
                memory_used = final_memory - initial_memory
                
                batch_time = end_time - start_time
                times.append(batch_time)
                memory_usage.append(memory_used)
                cpu_usage.append(np.mean(cpu_monitor) if cpu_monitor else 0)
                
                print(f"バッチ {i+1} 完了: {batch_time:.2f}秒, メモリ使用量: {memory_used:.1f}MB, CPU使用率: {np.mean(cpu_monitor):.1f}%")
            
            # 統計情報を計算
            results = {
                'implementation': 'parallel',
                'num_games': num_games,
                'num_batches': num_batches,
                'times': times,
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'total_time': sum(times),
                'memory_usage': memory_usage,
                'avg_memory': np.mean(memory_usage),
                'cpu_usage': cpu_usage,
                'avg_cpu': np.mean(cpu_usage),
                'games_per_second': num_games / np.mean(times),
                'parallel_efficiency': np.mean(cpu_usage) / 100 * self.system_info['cpu_count']
            }
            
            print(f"\n並列実装の結果:")
            print(f"  平均実行時間: {results['avg_time']:.2f}秒")
            print(f"  標準偏差: {results['std_time']:.2f}秒")
            print(f"  総実行時間: {results['total_time']:.2f}秒")
            print(f"  平均メモリ使用量: {results['avg_memory']:.1f}MB")
            print(f"  平均CPU使用率: {results['avg_cpu']:.1f}%")
            print(f"  ゲーム/秒: {results['games_per_second']:.2f}")
            print(f"  並列効率: {results['parallel_efficiency']:.2f}")
            
            return results
            
        except Exception as e:
            print(f"並列実装のベンチマークでエラー: {e}")
            traceback.print_exc()
            return None
    
    def run_comprehensive_benchmark(self):
        """包括的なベンチマークを実行"""
        print("包括的ベンチマークを開始します...")
        
        # マルチプロセス設定
        if multiprocessing.get_start_method() != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        
        # 各実装をベンチマーク
        benchmark_config = {
            'num_games': 8,  # 並列処理の効果を見るため
            'num_batches': 3
        }
        
        # 1. 元の実装
        original_results = self.benchmark_original_implementation(**benchmark_config)
        
        # 2. 並列実装
        parallel_results = self.benchmark_parallel_implementation(**benchmark_config)
        
        # 結果を保存
        self.results = {
            'original': original_results,
            'parallel': parallel_results,
            'system_info': self.system_info,
            'benchmark_config': benchmark_config,
            'timestamp': self.start_time.isoformat()
        }
        
        return self.results
    
    def generate_comparison_report(self):
        """比較レポートを生成"""
        print("\n" + "="*70)
        print("比較レポート")
        print("="*70)
        
        if not self.results:
            print("ベンチマーク結果がありません。")
            return
        
        # 有効な結果のみを使用
        valid_results = {k: v for k, v in self.results.items() if v is not None and k != 'system_info' and k != 'benchmark_config' and k != 'timestamp'}
        
        if len(valid_results) < 2:
            print("比較に十分な結果がありません。")
            return
        
        # 基準を設定（元の実装）
        if 'original' in valid_results:
            baseline = valid_results['original']
            baseline_name = '元の実装'
        else:
            print("基準となる実装が見つかりません。")
            return
        
        print(f"基準: {baseline_name}")
        print(f"基準性能: {baseline['avg_time']:.2f}秒/バッチ")
        print("-" * 70)
        
        # 比較表を作成
        comparison_data = []
        
        for impl_name, results in valid_results.items():
            if results is None:
                continue
                
            speedup = baseline['avg_time'] / results['avg_time']
            efficiency = (baseline['avg_time'] - results['avg_time']) / baseline['avg_time'] * 100
            
            comparison_data.append({
                'implementation': impl_name,
                'avg_time': results['avg_time'],
                'speedup': speedup,
                'efficiency': efficiency,
                'games_per_second': results['games_per_second'],
                'memory_usage': results['avg_memory'],
                'cpu_usage': results.get('avg_cpu', 'N/A')
            })
        
        # 結果をソート（スピードアップ順）
        comparison_data.sort(key=lambda x: x['speedup'], reverse=True)
        
        # 表形式で出力
        print(f"{'実装':<15} {'平均時間':<10} {'スピードアップ':<12} {'効率向上':<10} {'ゲーム/秒':<10} {'メモリ':<10} {'CPU使用率':<10}")
        print("-" * 80)
        
        for data in comparison_data:
            cpu_str = f"{data['cpu_usage']:.1f}%" if data['cpu_usage'] != 'N/A' else 'N/A'
            print(f"{data['implementation']:<15} {data['avg_time']:<10.2f} {data['speedup']:<12.2f} {data['efficiency']:<10.1f}% {data['games_per_second']:<10.2f} {data['memory_usage']:<10.1f} {cpu_str:<10}")
        
        # 最高性能の実装を特定
        best_impl = comparison_data[0]
        print(f"\n最高性能: {best_impl['implementation']}")
        print(f"  スピードアップ: {best_impl['speedup']:.2f}x")
        print(f"  効率向上: {best_impl['efficiency']:.1f}%")
        
        # 推奨事項
        print(f"\n推奨事項:")
        if best_impl['implementation'] == 'parallel':
            print("  - 並列処理版が最も効率的です")
            print("  - CPU使用率が高く、リソースを効率的に活用しています")
            print("  - 大規模な学習タスクに適しています")
        elif best_impl['implementation'] == 'original':
            print("  - 元の実装が安定した性能を提供します")
            print("  - メモリ使用量が少ない場合があります")
            print("  - 単純なタスクには十分な性能です")
        
        return comparison_data
    
    def save_results(self, filename='benchmark_results.json'):
        """結果をJSONファイルに保存"""
        if not self.results:
            print("保存する結果がありません。")
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n結果を {filename} に保存しました。")
        except Exception as e:
            print(f"結果の保存中にエラーが発生しました: {e}")
    
    def create_performance_graph(self):
        """パフォーマンスグラフを作成"""
        if not self.results:
            print("グラフを作成する結果がありません。")
            return
        
        try:
            # 有効な結果のみを使用
            valid_results = {k: v for k, v in self.results.items() if v is not None and k != 'system_info' and k != 'benchmark_config' and k != 'timestamp'}
            
            if len(valid_results) < 2:
                print("グラフを作成するには少なくとも2つの有効な結果が必要です。")
                return
            
            # グラフを作成
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            implementations = list(valid_results.keys())
            avg_times = [valid_results[impl]['avg_time'] for impl in implementations]
            games_per_second = [valid_results[impl]['games_per_second'] for impl in implementations]
            memory_usage = [valid_results[impl]['avg_memory'] for impl in implementations]
            
            # 実行時間の比較
            ax1.bar(implementations, avg_times, color=['blue', 'green', 'red'][:len(implementations)])
            ax1.set_title('平均実行時間の比較')
            ax1.set_ylabel('時間 (秒)')
            ax1.set_xlabel('実装')
            
            # ゲーム/秒の比較
            ax2.bar(implementations, games_per_second, color=['blue', 'green', 'red'][:len(implementations)])
            ax2.set_title('処理速度の比較')
            ax2.set_ylabel('ゲーム/秒')
            ax2.set_xlabel('実装')
            
            # メモリ使用量の比較
            ax3.bar(implementations, memory_usage, color=['blue', 'green', 'red'][:len(implementations)])
            ax3.set_title('メモリ使用量の比較')
            ax3.set_ylabel('メモリ使用量 (MB)')
            ax3.set_xlabel('実装')
            
            # スピードアップの比較
            if 'original' in valid_results:
                baseline = valid_results['original']['avg_time']
            else:
                baseline = max(avg_times)
            
            speedups = [baseline / time for time in avg_times]
            ax4.bar(implementations, speedups, color=['blue', 'green', 'red'][:len(implementations)])
            ax4.set_title('スピードアップの比較')
            ax4.set_ylabel('スピードアップ')
            ax4.set_xlabel('実装')
            ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig('alphazero_benchmark_comparison.png', dpi=300, bbox_inches='tight')
            print("パフォーマンスグラフを 'alphazero_benchmark_comparison.png' に保存しました。")
            
        except Exception as e:
            print(f"グラフの作成中にエラーが発生しました: {e}")
            traceback.print_exc()

def main():
    """メイン実行関数"""
    print("AlphaZero実装間の包括的ベンチマーク比較を開始します...")
    
    benchmark = ComprehensiveBenchmark()
    
    try:
        # ベンチマークを実行
        results = benchmark.run_comprehensive_benchmark()
        
        # 比較レポートを生成
        comparison_data = benchmark.generate_comparison_report()
        
        # 結果を保存
        benchmark.save_results()
        
        # パフォーマンスグラフを作成
        benchmark.create_performance_graph()
        
        print("\n" + "="*70)
        print("ベンチマーク完了!")
        print("="*70)
        print("以下のファイルが生成されました:")
        print("  - benchmark_results.json: 詳細な結果データ")
        print("  - alphazero_benchmark_comparison.png: パフォーマンスグラフ")
        
    except Exception as e:
        print(f"ベンチマーク実行中にエラーが発生しました: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
