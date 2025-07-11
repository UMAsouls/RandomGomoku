研究室用のランダム五目並べモジュールです。


# 学習の流れ
## DQNの学習
以下のプログラムを実行することでDQNの学習が行われます。
~/dqn_model/以下に保存されています。
``` 
python train.py
```

## 自分やagentと対戦させたい場合
対戦できますが、コードのコメントアウト等をいじって対戦相手を決定してください。
### vsHuman
```
python GUI.py
```
### agent vs agent
```
python play.py
```

# 各ファイルについて
```
.
├── GUI.py #人間がプレイする用 
├── GomokuEnv.py #学習環境です　評価＆報酬関数等あるので高頻度でさわると思います
├── README.md
├── RandomGomoku #五目並べゲーム本体
│   ├── Board #学習コード書くときは基本ここ見ておけばおｋ
│   │   ├── Board.py
│   │   ├── RandomSetter.py
│   │   └── __init__.py
│   ├── Dependency.py
│   ├── GetBoard.py
│   ├── Interfaces
│   │   ├── CreatingMass.py
│   │   ├── IHeadMass.py
│   │   ├── IMass.py
│   │   └── __init__.py
│   ├── Mass
│   │   ├── HeadMass.py
│   │   ├── Mass.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── __main__.py
│   └── const
│       ├── Stone.py #石の情報がまとめられているのでたまに必要
│       └── __init__.py
├── agent.py #RandomAgentがデフォで入ってます。他のエージェントも必要ならここに書くと綺麗かも
├── dqn.py # dqn周りは基本ここ　いじる頻度多め
├── dqn_model # dqnの学習モデルが保存される
│   └── qnet.npz
├── function.md # RandomGomokuの基本的な関数がまとめられてる
├── main.py # RandomGomokuを試すためのファイル
├── play.py # 作ったagent同士が戦うためのファイル
├── pyproject.toml
└── train.py　# 学習実行用
```

# AlphaZero五目並べ

PyTorchを使用したAlphaZeroアルゴリズムによる五目並べAIの実装です。

## 必要な環境

- Python 3.7以上
- PyTorch
- NumPy
- Matplotlib
- tqdm
- numba

## インストール

```bash
# 必要なパッケージのインストール
pip install torch numpy matplotlib tqdm numba
```

## 使用方法

### 1. バッチファイルを使用（Windows）

#### 訓練の実行
```cmd
run_training.bat
```

#### 人間との対戦
```cmd
run_play.bat
```

### 2. Pythonスクリプトを直接実行

#### 基本的な訓練
```bash
python run_alpha_gomoku.py --train
```

#### 高速テスト（小さい設定）
```bash
python run_alpha_gomoku.py --train --quick
```

#### カスタム設定での訓練
```bash
python run_alpha_gomoku.py --train --board-size 15 --iterations 100 --self-play-games 64
```

#### 人間との対戦
```bash
python run_alpha_gomoku.py --play --board-size 8
```

### 3. 元のスクリプトを直接実行
```bash
python alpha_gomoku.py --board-size 8 --iterations 100 --play
```

## パラメータ説明

- `--board-size`: 盤面のサイズ（デフォルト: 8）
- `--iterations`: 訓練イテレーション数（デフォルト: 10000）
- `--self-play-games`: イテレーションごとの自己対戦ゲーム数（デフォルト: 192）
- `--lr`: 初期学習率（デフォルト: 0.0002）
- `--quick`: 高速テスト設定を使用
- `--model-dir`: モデル保存ディレクトリ（デフォルト: models）
- `--log-dir`: ログ保存ディレクトリ（デフォルト: logs）

## ディレクトリ構成

```
RandomGomoku/
├── alpha_gomoku.py          # メインのAlphaZero実装
├── run_alpha_gomoku.py      # 実行用スクリプト
├── run_training.bat         # 訓練用バッチファイル（Windows）
├── run_play.bat            # 対戦用バッチファイル（Windows）
├── GomokuEnv.py            # 五目並べ環境
├── models/                 # 訓練済みモデル保存先
├── logs/                   # 訓練ログ保存先
└── README.md              # このファイル
```

## 対戦の操作方法

1. 先手・後手を選択
2. 列番号と行番号を入力（0から始まる）
3. 5つ並べると勝利

例：
```
列 (0-7): 3
行 (0-7): 4
```

## 訓練について

- 訓練には長時間かかります（数時間〜数日）
- `--quick`オプションで高速テストが可能
- 訓練中のログとグラフは`logs/`ディレクトリに保存
- モデルは`models/`ディレクトリに保存

## トラブルシューティング

### CUDA使用時のメモリエラー
```bash
# バッチサイズを小さくする
python alpha_gomoku.py --train --self-play-games 32
```

### 訓練が遅い場合
```bash
# 高速設定を使用
python run_alpha_gomoku.py --train --quick
```

### パッケージが見つからない
```bash
pip install torch numpy matplotlib tqdm numba
```
