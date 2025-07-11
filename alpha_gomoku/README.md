# AlphaGomoku - 構造化版

AlphaZeroアルゴリズムを使用した五目並べAIの実装です。

## 新しい構造

```
alpha_gomoku/
├── __init__.py              # パッケージ初期化
├── config.py                # 設定ファイル
├── models/                  # モデル関連
│   ├── __init__.py
│   └── dual_network.py      # ニューラルネットワーク
├── mcts/                    # MCTS関連
│   ├── __init__.py
│   ├── node.py             # MCTSノード
│   ├── mcts.py             # MCTS実装
│   └── game_utils.py       # ゲームルールユーティリティ
├── training/                # トレーニング関連
│   ├── __init__.py
│   ├── replay_buffer.py     # 経験リプレイバッファ
│   ├── self_play.py        # 自己対戦データセット
│   ├── train_network.py    # ネットワーク訓練
│   └── trainer.py          # AlphaZeroトレーナー
├── utils/                   # ユーティリティ
│   ├── __init__.py
│   └── helpers.py          # ヘルパー関数
└── legacy_alpha_gomoku.py   # 元のファイル
```

## 主な改善点

### 1. モジュール分割
- **models/**: ニューラルネットワーク関連
- **mcts/**: モンテカルロ木探索関連
- **training/**: 訓練プロセス関連
- **utils/**: 共通ユーティリティ

### 2. 保守性の向上
- 各機能が独立したモジュールに分離
- 責任の明確化
- テストしやすい構造

### 3. 拡張性の向上
- 新しい機能の追加が容易
- 既存コードへの影響を最小化
- 設定の中央管理

## 使用方法

### トレーニング
```bash
python train_alpha_gomoku_new.py --board_size 8 --iterations 100 --games 32
```

### 人間との対戦
```bash
python play_alpha_gomoku_new.py --board_size 8
```

### コマンドライン引数
- `--board_size`: 盤面サイズ (デフォルト: 8)
- `--iterations`: 訓練イテレーション数 (デフォルト: 100)
- `--games`: 各イテレーションの自己対戦ゲーム数 (デフォルト: 32)
- `--lr`: 学習率 (デフォルト: 0.001)
- `--checkpoint_dir`: モデル保存ディレクトリ (デフォルト: models)
- `--log_dir`: ログディレクトリ (デフォルト: logs)

## 元のファイル

元の`alpha_gomoku.py`は`alpha_gomoku/legacy_alpha_gomoku.py`に移動されました。
