"""
設定ファイル
"""

import torch

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ゲーム設定
DEFAULT_BOARD_SIZE = 8  # デフォルトのボードサイズ
SIMULATIONS = 300  # MCTSのシミュレーション回数

# トレーニング設定
DEFAULT_ITERATIONS = 5000  # デフォルトのトレーニングイテレーション数
DEFAULT_SELF_PLAY_GAMES = 800  # イテレーションごとの自己対戦ゲーム数
DEFAULT_LEARNING_RATE = 0.001  # 初期学習率

# MCTS設定
MCTS_NUM_SIMULATIONS = 300
MCTS_C_PUCT = 4.0
MCTS_USE_GUMBEL = False
MCTS_GUMBEL_SCALE = 0.1
MCTS_ADD_ROOT_NOISE = True
MCTS_DIRICHLET_ALPHA = 0.15
MCTS_DIRICHLET_WEIGHT = 0.25
MCTS_TEMPERATURE_THRESHOLD = 30
MCTS_INITIAL_TEMPERATURE = 1.0
MCTS_FINAL_TEMPERATURE = 1e-3

# ネットワーク設定
NETWORK_NUM_CHANNELS = 32  # 初期チャネル数
NETWORK_NUM_RES_BLOCKS = 3  # 残差ブロック数
NETWORK_DROPOUT_RATE = 0.3  # ドロップアウト率

# トレーニング設定（詳細）
TRAINING_EPOCHS = 20  # 訓練エポック数（増加）
TRAINING_BATCH_SIZE = 64  # バッチサイズ（減少して安定化）
TRAINING_WEIGHT_DECAY = 1e-4  # 重み減衰
TRAINING_LR_STEP_SIZE = 10  # 学習率ステップサイズ（緩やか）
TRAINING_LR_GAMMA = 0.95  # 学習率減衰率（より緩やか）
TRAINING_MAX_SAMPLES = 10000  # 最大サンプル数（適切な範囲に調整）
TRAINING_PATIENCE = 8  # 早期停止のpatience（増加）
TRAINING_NUM_WORKERS = 2  # データローダーのワーカー数
TRAINING_LR_DECAY_FACTOR = 0.7  # 学習率減衰係数（緩やか）

# リプレイバッファ設定
REPLAY_BUFFER_CAPACITY = 50000  # リプレイバッファ容量（増加）

# 自己対戦設定
SELF_PLAY_RESULT_WEIGHT_MIN = 0.2  # 結果重み最小値
SELF_PLAY_RESULT_WEIGHT_MAX = 0.8  # 結果重み最大値
SELF_PLAY_RESULT_WEIGHT_FACTOR = 0.6  # 結果重み係数

# AlphaZeroトレーナー設定
TRAINER_BOARD_SIZE = 8  # ボードサイズ
TRAINER_NUM_ITERATIONS = 100  # イテレーション数
TRAINER_NUM_SELF_PLAY_GAMES = 128  # 自己対戦ゲーム数（増加）
TRAINER_INITIAL_LR = 0.0001  # 初期学習率（より低く設定）
TRAINER_CHECKPOINT_DIR = 'models'  # チェックポイントディレクトリ
TRAINER_LOG_DIR = 'logs'  # ログディレクトリ


class Config:
    """設定管理クラス"""
    
    def __init__(self):
        # デバイス設定
        self.device = device
        
        # ゲーム設定
        self.board_size = DEFAULT_BOARD_SIZE
        self.simulations = SIMULATIONS
        
        # MCTS設定
        self.mcts_num_simulations = MCTS_NUM_SIMULATIONS
        self.mcts_c_puct = MCTS_C_PUCT
        self.mcts_use_gumbel = MCTS_USE_GUMBEL
        self.mcts_gumbel_scale = MCTS_GUMBEL_SCALE
        self.mcts_add_root_noise = MCTS_ADD_ROOT_NOISE
        self.mcts_dirichlet_alpha = MCTS_DIRICHLET_ALPHA
        self.mcts_dirichlet_weight = MCTS_DIRICHLET_WEIGHT
        self.mcts_temperature_threshold = MCTS_TEMPERATURE_THRESHOLD
        self.mcts_initial_temperature = MCTS_INITIAL_TEMPERATURE
        self.mcts_final_temperature = MCTS_FINAL_TEMPERATURE
        
        # ネットワーク設定
        self.network_num_channels = NETWORK_NUM_CHANNELS
        self.network_num_res_blocks = NETWORK_NUM_RES_BLOCKS
        self.network_dropout_rate = NETWORK_DROPOUT_RATE
        
        # トレーニング設定
        self.training_epochs = TRAINING_EPOCHS
        self.training_batch_size = TRAINING_BATCH_SIZE
        self.training_lr = DEFAULT_LEARNING_RATE
        self.training_weight_decay = TRAINING_WEIGHT_DECAY
        self.training_lr_step_size = TRAINING_LR_STEP_SIZE
        self.training_lr_gamma = TRAINING_LR_GAMMA
        self.training_max_samples = TRAINING_MAX_SAMPLES
        self.training_patience = TRAINING_PATIENCE
        self.training_num_workers = TRAINING_NUM_WORKERS
        self.training_lr_decay_factor = TRAINING_LR_DECAY_FACTOR
        
        # リプレイバッファ設定
        self.replay_buffer_capacity = REPLAY_BUFFER_CAPACITY
        
        # 自己対戦設定
        self.self_play_result_weight_min = SELF_PLAY_RESULT_WEIGHT_MIN
        self.self_play_result_weight_factor = SELF_PLAY_RESULT_WEIGHT_FACTOR
        
        # AlphaZeroトレーナー設定
        self.trainer_board_size = TRAINER_BOARD_SIZE
        self.trainer_num_iterations = TRAINER_NUM_ITERATIONS
        self.trainer_num_self_play_games = TRAINER_NUM_SELF_PLAY_GAMES
        self.trainer_initial_lr = TRAINER_INITIAL_LR
        self.trainer_checkpoint_dir = TRAINER_CHECKPOINT_DIR
        self.trainer_log_dir = TRAINER_LOG_DIR
    
    def update_from_dict(self, config_dict):
        """辞書からパラメータを更新"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self):
        """パラメータを辞書として返す"""
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
    
    def save_config(self, filepath):
        """設定をファイルに保存"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_config(cls, filepath):
        """ファイルから設定を読み込み"""
        import json
        config = cls()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config.update_from_dict(config_dict)
        except FileNotFoundError:
            print(f"設定ファイル {filepath} が見つかりません。デフォルト設定を使用します。")
        return config


# デフォルト設定インスタンス
default_config = Config()
