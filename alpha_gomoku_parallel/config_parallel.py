"""
並列処理版AlphaZeroの設定ファイル
"""

# ボードゲーム設定
BOARD_SIZE = 8
N_IN_ROW = 5

# 並列処理設定
PARALLEL_CONFIG = {
    # 自動でCPUコア数を検出して設定（-1は学習用に予約）
    'auto_detect_workers': True,
    
    # 手動設定（auto_detect_workers=Falseの場合に使用）
    'manual_game_workers': 4,
    'manual_data_workers': 2,
    
    # 並列処理の制限
    'max_game_workers': 8,
    'max_data_workers': 4,
    
    # GPU使用設定
    'gpu_memory_fraction': 0.8,  # GPU使用率の制限
    'gpu_batch_size': 32,
    
    # メモリ管理
    'memory_threshold': 80,  # メモリ使用率の閾値（%）
    'adaptive_batch_size': True,  # バッチサイズの動的調整
}

# トレーニング設定
TRAINING_CONFIG = {
    'learn_rate': 2e-3,
    'lr_multiplier': 1.0,
    'temp': 1.0,
    'n_playout': 400,
    'c_puct': 5,
    'buffer_size': 10000,
    'batch_size': 512,
    'epochs': 20,
    'kl_targ': 0.02,
    'check_freq': 100,
    'game_batch_num': 5000,
}

# パフォーマンス監視設定
MONITORING_CONFIG = {
    'enable_performance_monitoring': True,
    'log_interval': 10,  # ログ出力間隔
    'save_performance_graph': True,
    'performance_log_file': 'performance_log.txt',
}

# ファイル設定
FILE_CONFIG = {
    'best_model_path': './best_policy_parallel.model',
    'current_model_path': './current_policy_parallel.model',
    'loss_graph_path': './loss_graph_parallel.png',
    'performance_graph_path': './performance_graph_parallel.png',
}

# デバッグ設定
DEBUG_CONFIG = {
    'verbose': True,
    'show_game_progress': False,
    'save_game_logs': False,
    'benchmark_mode': False,
}
