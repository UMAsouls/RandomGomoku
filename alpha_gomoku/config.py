"""
設定ファイル
"""

import torch

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ゲーム設定
DEFAULT_BOARD_SIZE = 8  # デフォルトのボードサイズ
SIMULATIONS = 200  # MCTSのシミュレーション回数

# トレーニング設定
DEFAULT_ITERATIONS = 5000  # デフォルトのトレーニングイテレーション数
DEFAULT_SELF_PLAY_GAMES = 32  # イテレーションごとの自己対戦ゲーム数
DEFAULT_LEARNING_RATE = 0.001  # 初期学習率
