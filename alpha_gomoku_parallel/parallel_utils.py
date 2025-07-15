"""
並列処理用のユーティリティクラス
CPU/GPU リソースの管理と最適化を行います
"""
import multiprocessing
import torch
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class ResourceManager:
    """
    システムリソースを管理し、最適な並列処理戦略を決定するクラス
    """
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU情報を取得
        self.gpu_memory = []
        if self.gpu_available:
            for i in range(self.gpu_count):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.gpu_memory.append(gpu_memory)
        
        self._print_system_info()
        self._determine_optimal_strategy()
    
    def _print_system_info(self):
        """システム情報を表示"""
        print("=" * 50)
        print("システムリソース情報:")
        print(f"  CPU コア数: {self.cpu_count}")
        print(f"  システムメモリ: {self.memory_gb:.1f} GB")
        print(f"  GPU利用可能: {self.gpu_available}")
        if self.gpu_available:
            for i, mem in enumerate(self.gpu_memory):
                print(f"  GPU {i} メモリ: {mem:.1f} GB")
        print("=" * 50)
    
    def _determine_optimal_strategy(self):
        """最適な並列処理戦略を決定"""
        # 基本的な並列処理パラメータ
        self.optimal_game_workers = min(self.cpu_count - 1, 8)  # 学習用に1コア残す
        self.optimal_data_workers = min(self.cpu_count // 2, 4)
        
        # GPU使用時の調整
        if self.gpu_available:
            self.optimal_training_batch_size = 64
            self.optimal_gpu_batch_size = 32
            self.use_gpu_for_training = True
        else:
            self.optimal_training_batch_size = 32
            self.optimal_gpu_batch_size = 16
            self.use_gpu_for_training = False
        
        # メモリ使用量による調整
        if self.memory_gb < 8:
            self.optimal_game_workers = min(self.optimal_game_workers, 4)
            self.optimal_training_batch_size = min(self.optimal_training_batch_size, 32)
        
        print("最適化された並列処理パラメータ:")
        print(f"  並列ゲーム数: {self.optimal_game_workers}")
        print(f"  データ処理ワーカー数: {self.optimal_data_workers}")
        print(f"  学習バッチサイズ: {self.optimal_training_batch_size}")
        print(f"  GPU使用: {self.use_gpu_for_training}")
        print("=" * 50)
    
    def get_optimal_params(self):
        """最適化されたパラメータを返す"""
        return {
            'game_workers': self.optimal_game_workers,
            'data_workers': self.optimal_data_workers,
            'training_batch_size': self.optimal_training_batch_size,
            'gpu_batch_size': self.optimal_gpu_batch_size,
            'use_gpu': self.use_gpu_for_training
        }

class PerformanceMonitor:
    """
    パフォーマンスを監視するクラス
    """
    
    def __init__(self):
        self.game_times = []
        self.training_times = []
        self.memory_usage = []
        self.gpu_usage = []
        self.start_time = time.time()
        
    def record_game_time(self, duration):
        """ゲーム実行時間を記録"""
        self.game_times.append(duration)
        
    def record_training_time(self, duration):
        """学習時間を記録"""
        self.training_times.append(duration)
        
    def record_memory_usage(self):
        """メモリ使用量を記録"""
        memory_percent = psutil.virtual_memory().percent
        self.memory_usage.append(memory_percent)
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.gpu_usage.append(gpu_memory)
    
    def print_performance_summary(self):
        """パフォーマンスサマリーを表示"""
        if not self.game_times:
            return
            
        total_time = time.time() - self.start_time
        avg_game_time = sum(self.game_times) / len(self.game_times)
        avg_training_time = sum(self.training_times) / len(self.training_times) if self.training_times else 0
        
        print("\n" + "=" * 50)
        print("パフォーマンスサマリー:")
        print(f"  総実行時間: {total_time:.2f}秒")
        print(f"  平均ゲーム時間: {avg_game_time:.3f}秒")
        print(f"  平均学習時間: {avg_training_time:.3f}秒")
        print(f"  総ゲーム数: {len(self.game_times)}")
        print(f"  総学習回数: {len(self.training_times)}")
        
        if self.memory_usage:
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            print(f"  平均メモリ使用率: {avg_memory:.1f}%")
        
        if self.gpu_usage:
            avg_gpu = sum(self.gpu_usage) / len(self.gpu_usage)
            print(f"  平均GPU使用率: {avg_gpu:.1f}%")
        
        print("=" * 50)

class ParallelCoordinator:
    """
    並列処理を調整するクラス
    """
    
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.params = resource_manager.get_optimal_params()
        self.monitor = PerformanceMonitor()
        
        # 並列処理用のExecutorを初期化
        self.game_executor = None
        self.data_executor = None
        self.training_lock = threading.Lock()
        
    def initialize_executors(self):
        """Executorを初期化"""
        self.game_executor = ProcessPoolExecutor(max_workers=self.params['game_workers'])
        self.data_executor = ThreadPoolExecutor(max_workers=self.params['data_workers'])
        
    def cleanup_executors(self):
        """Executorをクリーンアップ"""
        if self.game_executor:
            self.game_executor.shutdown(wait=True)
        if self.data_executor:
            self.data_executor.shutdown(wait=True)
            
    def get_performance_monitor(self):
        """パフォーマンスモニターを取得"""
        return self.monitor
    
    def optimize_batch_size(self, current_batch_size, memory_usage):
        """メモリ使用量に基づいてバッチサイズを調整"""
        if memory_usage > 80:  # 80%以上の場合
            return max(current_batch_size // 2, 16)
        elif memory_usage < 50:  # 50%以下の場合
            return min(current_batch_size * 2, 1024)
        else:
            return current_batch_size
