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
```
python play.py
```

# 各ファイルについて
```
.
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
├── play.py # つくったagent等と戦うためのファイル
├── pyproject.toml
└── train.py　# 学習実行用
```
