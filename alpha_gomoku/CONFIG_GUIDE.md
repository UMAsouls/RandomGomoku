# AlphaGomoku 設定ガイド

## 概要
alpha_gomokuディレクトリ内のすべてのパラメータがconfig.pyファイルで管理できるようになりました。

## 設定ファイル (config.py)

### 基本設定
- `DEFAULT_BOARD_SIZE`: ボードサイズ (デフォルト: 8)
- `SIMULATIONS`: MCTSシミュレーション回数 (デフォルト: 200)
- `DEFAULT_LEARNING_RATE`: 学習率 (デフォルト: 0.001)

### MCTS設定
- `MCTS_NUM_SIMULATIONS`: MCTSシミュレーション回数 (デフォルト: 200)
- `MCTS_C_PUCT`: UCTの定数 (デフォルト: 4.0)
- `MCTS_USE_GUMBEL`: Gumbel分布使用フラグ (デフォルト: False)
- `MCTS_GUMBEL_SCALE`: Gumbelスケール (デフォルト: 0.1)
- `MCTS_ADD_ROOT_NOISE`: ルートノイズ追加フラグ (デフォルト: True)
- `MCTS_DIRICHLET_ALPHA`: Dirichlet分布のα (デフォルト: 0.15)
- `MCTS_DIRICHLET_WEIGHT`: Dirichletノイズの重み (デフォルト: 0.25)
- `MCTS_TEMPERATURE_THRESHOLD`: 温度閾値 (デフォルト: 30)
- `MCTS_INITIAL_TEMPERATURE`: 初期温度 (デフォルト: 1.0)
- `MCTS_FINAL_TEMPERATURE`: 最終温度 (デフォルト: 1e-3)

### ネットワーク設定
- `NETWORK_NUM_CHANNELS`: 初期チャネル数 (デフォルト: 32)
- `NETWORK_NUM_RES_BLOCKS`: 残差ブロック数 (デフォルト: 3)
- `NETWORK_DROPOUT_RATE`: ドロップアウト率 (デフォルト: 0.3)

### トレーニング設定
- `TRAINING_EPOCHS`: 訓練エポック数 (デフォルト: 10)
- `TRAINING_BATCH_SIZE`: バッチサイズ (デフォルト: 256)
- `TRAINING_WEIGHT_DECAY`: 重み減衰 (デフォルト: 1e-4)
- `TRAINING_LR_STEP_SIZE`: 学習率ステップサイズ (デフォルト: 3)
- `TRAINING_LR_GAMMA`: 学習率減衰率 (デフォルト: 0.8)
- `TRAINING_MAX_SAMPLES`: 最大サンプル数 (デフォルト: 3000)
- `TRAINING_PATIENCE`: 早期停止のpatience (デフォルト: 2)
- `TRAINING_NUM_WORKERS`: データローダーのワーカー数 (デフォルト: 2)
- `TRAINING_LR_DECAY_FACTOR`: 学習率減衰係数 (デフォルト: 0.5)

### リプレイバッファ設定
- `REPLAY_BUFFER_CAPACITY`: リプレイバッファ容量 (デフォルト: 40000)

### 自己対戦設定
- `SELF_PLAY_RESULT_WEIGHT_MIN`: 結果重み最小値 (デフォルト: 0.2)
- `SELF_PLAY_RESULT_WEIGHT_FACTOR`: 結果重み係数 (デフォルト: 0.6)

### AlphaZeroトレーナー設定
- `TRAINER_BOARD_SIZE`: ボードサイズ (デフォルト: 19)
- `TRAINER_NUM_ITERATIONS`: イテレーション数 (デフォルト: 100)
- `TRAINER_NUM_SELF_PLAY_GAMES`: 自己対戦ゲーム数 (デフォルト: 64)
- `TRAINER_INITIAL_LR`: 初期学習率 (デフォルト: 0.0005)
- `TRAINER_CHECKPOINT_DIR`: チェックポイントディレクトリ (デフォルト: 'models')
- `TRAINER_LOG_DIR`: ログディレクトリ (デフォルト: 'logs')

## 使用方法

### 基本的な使用
```python
from alpha_gomoku.config import default_config

# デフォルト設定を使用
trainer = AlphaZero(
    board_size=default_config.trainer_board_size,
    num_iterations=default_config.trainer_num_iterations,
    num_self_play_games=default_config.trainer_num_self_play_games
)
```

### 設定のカスタマイズ
```python
from alpha_gomoku.config import Config

# カスタム設定を作成
config = Config()
config.trainer_board_size = 15
config.trainer_num_iterations = 200
config.mcts_num_simulations = 500

# 設定を保存
config.save_config('custom_config.json')

# 設定を読み込み
config = Config.load_config('custom_config.json')
```

### 辞書からの設定更新
```python
from alpha_gomoku.config import Config

config = Config()
custom_params = {
    'trainer_board_size': 11,
    'mcts_num_simulations': 300,
    'network_num_channels': 64
}
config.update_from_dict(custom_params)
```

## 設定ファイルのJSON形式
```json
{
  "trainer_board_size": 15,
  "trainer_num_iterations": 200,
  "mcts_num_simulations": 500,
  "network_num_channels": 64,
  "training_batch_size": 128
}
```

## 注意事項
- 設定を変更する際は、メモリ使用量とトレーニング時間への影響を考慮してください
- バッチサイズが大きすぎるとメモリ不足を起こす可能性があります
- MCTSシミュレーション回数を増やすと計算時間が増加します
- ネットワークのチャネル数や残差ブロック数を増やすとモデルが複雑になります

## 推奨設定
### 小さなボード（8x8）
```python
config.trainer_board_size = 8
config.mcts_num_simulations = 200
config.network_num_channels = 32
config.training_batch_size = 256
```

### 中程度のボード（15x15）
```python
config.trainer_board_size = 15
config.mcts_num_simulations = 400
config.network_num_channels = 64
config.training_batch_size = 128
```

### 大きなボード（19x19）
```python
config.trainer_board_size = 19
config.mcts_num_simulations = 800
config.network_num_channels = 128
config.training_batch_size = 64
```
