"""
さらなる学習安定化のための設定
"""

# より安定した学習のための推奨設定

# 1. バッチ正規化を有効にする
ENABLE_BATCH_NORM = True

# 2. 温かい再開（Warm Restart）のための設定
COSINE_ANNEALING_WARM_RESTART = True
COSINE_ANNEALING_T_0 = 10  # 初期サイクル長
COSINE_ANNEALING_T_MULT = 2  # サイクル倍率

# 3. 損失の平滑化設定
LOSS_SMOOTHING_FACTOR = 0.1  # 指数移動平均の係数

# 4. 学習率の範囲制限
MIN_LEARNING_RATE = 1e-7
MAX_LEARNING_RATE = 1e-3

# 5. 勾配の累積（Gradient Accumulation）
GRADIENT_ACCUMULATION_STEPS = 2

# 6. より高度な正則化
LABEL_SMOOTHING = 0.1  # ラベル平滑化
MIXUP_ALPHA = 0.2  # MixUp 正則化

# 7. 学習状態の監視
MONITOR_GRADIENT_NORM = True
GRADIENT_NORM_THRESHOLD = 2.0

# 8. データの品質改善
FILTER_LOW_QUALITY_GAMES = True
MIN_GAME_LENGTH = 10  # 最小ゲーム長
MAX_GAME_LENGTH = 100  # 最大ゲーム長

# 9. より安定したMCTSパラメータ
STABLE_MCTS_C_PUCT = 2.0  # より保守的な探索
STABLE_MCTS_TEMPERATURE = 0.5  # より決定論的な行動選択

# 10. 学習データのバランス調整
BALANCED_SAMPLING = True
WIN_LOSS_RATIO_TARGET = 0.5  # 勝敗データの比率目標

# 11. ログ出力の設定
LOG_EPOCH_DETAILS = True  # エポック毎の詳細ログを出力
LOG_ITERATION_SUMMARY = True  # イテレーション毎の要約ログを出力
SAVE_EPOCH_LOGS = True  # エポック毎のログをファイルに保存

# 12. 棋譜保存の設定
SAVE_GAME_RECORDS = True  # 棋譜の保存を有効にする
SAVE_GAME_RECORDS_PER_ITERATION = 1  # イテレーション毎に保存する棋譜数
GAME_RECORD_FORMAT = 'PGN'  # 棋譜の保存形式 ('PGN', 'JSON', 'TEXT')
SAVE_DETAILED_GAME_INFO = True  # 詳細なゲーム情報も保存
