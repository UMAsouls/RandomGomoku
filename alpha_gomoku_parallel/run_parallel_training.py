"""
並列処理版AlphaZeroの実行スクリプト
"""
import multiprocessing
import sys
import os
import torch
from alphazero_parallel import AlphaZeroParallel
from parallel_utils import ResourceManager, ParallelCoordinator

def main():
    # マルチプロセス用の設定
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    print("並列処理版AlphaZeroを開始します...")
    
    # デバイス情報を表示
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # リソースマネージャーを初期化
    resource_manager = ResourceManager()
    
    # 並列処理コーディネーターを初期化
    coordinator = ParallelCoordinator(resource_manager)
    
    try:
        # 並列処理版AlphaZeroを実行
        alpha_zero = AlphaZeroParallel()
        alpha_zero.train()
        
    except KeyboardInterrupt:
        print("\n\rユーザーによる中断を検出しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # リソースのクリーンアップ
        coordinator.cleanup_executors()
        
        # パフォーマンスサマリーを表示
        monitor = coordinator.get_performance_monitor()
        monitor.print_performance_summary()
        
        print("並列処理版AlphaZeroを終了しました。")

if __name__ == "__main__":
    main()
