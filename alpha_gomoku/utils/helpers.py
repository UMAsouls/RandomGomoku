"""
ユーティリティ関数
"""

import argparse
import os
import datetime
from ..config import (
    DEFAULT_BOARD_SIZE, DEFAULT_ITERATIONS, DEFAULT_SELF_PLAY_GAMES,
    DEFAULT_LEARNING_RATE, TRAINER_CHECKPOINT_DIR, TRAINER_LOG_DIR
)


def get_args():
    """コマンドライン引数を取得"""
    parser = argparse.ArgumentParser(description='AlphaGomoku Training')
    parser.add_argument('--board_size', type=int, default=DEFAULT_BOARD_SIZE, help=f'Board size (default: {DEFAULT_BOARD_SIZE})')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS, help=f'Training iterations (default: {DEFAULT_ITERATIONS})')
    parser.add_argument('--games', type=int, default=DEFAULT_SELF_PLAY_GAMES, help=f'Self-play games per iteration (default: {DEFAULT_SELF_PLAY_GAMES})')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help=f'Learning rate (default: {DEFAULT_LEARNING_RATE})')
    parser.add_argument('--checkpoint_dir', type=str, default=TRAINER_CHECKPOINT_DIR, help=f'Checkpoint directory (default: {TRAINER_CHECKPOINT_DIR})')
    parser.add_argument('--log_dir', type=str, default=TRAINER_LOG_DIR, help=f'Log directory (default: {TRAINER_LOG_DIR})')
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
