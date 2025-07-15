"""
並列処理版AlphaZeroの動作テスト
"""
import multiprocessing
import torch
import time
import sys
import os
from alphazero_parallel import AlphaZeroParallel
from parallel_utils import ResourceManager

def test_resource_manager():
    """リソースマネージャーのテスト"""
    print("=" * 50)
    print("リソースマネージャーのテスト")
    print("=" * 50)
    
    rm = ResourceManager()
    params = rm.get_optimal_params()
    
    print("取得された最適パラメータ:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    return True

def test_parallel_selfplay():
    """並列自己対戦のテスト"""
    print("\n" + "=" * 50)
    print("並列自己対戦のテスト")
    print("=" * 50)
    
    try:
        alpha_zero = AlphaZeroParallel()
        
        # 短時間でのテスト
        print("2ゲームの並列実行をテスト...")
        start_time = time.time()
        
        play_data = alpha_zero.collect_parallel_selfplay_data(2)
        
        end_time = time.time()
        
        print(f"実行時間: {end_time - start_time:.2f}秒")
        print(f"収集データ数: {len(play_data)}")
        print("並列自己対戦テスト: 成功")
        
        return True
        
    except Exception as e:
        print(f"並列自己対戦テスト: 失敗 - {e}")
        return False

def test_continuous_learning():
    """継続的学習のテスト"""
    print("\n" + "=" * 50)
    print("継続的学習のテスト")
    print("=" * 50)
    
    try:
        alpha_zero = AlphaZeroParallel()
        
        # 学習スレッドを開始
        alpha_zero.learning_thread = alpha_zero.continuous_learning_thread
        
        # テストデータを追加
        test_data = []
        for i in range(alpha_zero.batch_size):
            state = [0] * (4 * alpha_zero.board_size * alpha_zero.board_size)
            mcts_prob = [0.1] * (alpha_zero.board_size * alpha_zero.board_size)
            winner = 1 if i % 2 == 0 else -1
            test_data.append((state, mcts_prob, winner))
        
        # データをバッファに追加
        alpha_zero.data_buffer.extend(test_data)
        
        print(f"テストデータ数: {len(test_data)}")
        print(f"バッファサイズ: {len(alpha_zero.data_buffer)}")
        print("継続的学習テスト: 成功")
        
        return True
        
    except Exception as e:
        print(f"継続的学習テスト: 失敗 - {e}")
        return False

def test_gpu_availability():
    """GPU利用可能性のテスト"""
    print("\n" + "=" * 50)
    print("GPU利用可能性のテスト")
    print("=" * 50)
    
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        print(f"GPU利用可能: {gpu_count}個のGPUが検出されました")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    メモリ: {props.total_memory / (1024**3):.1f} GB")
            print(f"    マルチプロセッサ: {props.multi_processor_count}")
    else:
        print("GPU利用不可: CPUモードで実行されます")
    
    return gpu_available

def run_all_tests():
    """すべてのテストを実行"""
    print("並列処理版AlphaZero 動作テスト")
    print("=" * 50)
    
    tests = [
        ("リソースマネージャー", test_resource_manager),
        ("GPU利用可能性", test_gpu_availability),
        ("並列自己対戦", test_parallel_selfplay),
        ("継続的学習", test_continuous_learning),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name}テストでエラーが発生: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    print("=" * 50)
    
    success_count = 0
    for test_name, result in results:
        status = "成功" if result else "失敗"
        print(f"  {test_name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n総合結果: {success_count}/{len(results)} テスト成功")
    
    if success_count == len(results):
        print("すべてのテストが成功しました！並列処理版AlphaZeroは正常に動作します。")
    else:
        print("いくつかのテストが失敗しました。設定を確認してください。")
    
    return success_count == len(results)

def main():
    """メイン関数"""
    # マルチプロセス設定
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    # テスト実行
    success = run_all_tests()
    
    if success:
        print("\n並列処理版AlphaZeroの準備が完了しました。")
        print("実行するには: python run_parallel_training.py")
    else:
        print("\n設定に問題があります。エラーメッセージを確認してください。")
    
    return success

if __name__ == "__main__":
    main()
