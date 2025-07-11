"""
ユーティリティ関数
"""

import argparse
import os
import datetime


def get_args():
    """コマンドライン引数を取得"""
    parser = argparse.ArgumentParser(description='AlphaGomoku Training')
    parser.add_argument('--board_size', type=int, default=8, help='Board size (default: 8)')
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations (default: 100)')
    parser.add_argument('--games', type=int, default=32, help='Self-play games per iteration (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--checkpoint_dir', type=str, default='models', help='Checkpoint directory (default: models)')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory (default: logs)')
    return parser.parse_args()


def create_timestamp():
    """現在時刻のタイムスタンプを作成"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs(*dirs):
    """ディレクトリを作成"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def find_latest_model(checkpoint_dir, model_base_name):
    """最新のモデルファイルを検索"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    model_files = [f for f in os.listdir(checkpoint_dir) 
                  if f.startswith(model_base_name) and f.endswith('.pth')]
    
    if not model_files:
        return None
    
    # 最新のモデルを返す
    return os.path.join(checkpoint_dir, sorted(model_files)[-1])
