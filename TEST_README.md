# AlphaGomoku Test Files

## 作成されたテストファイル

### 1. `train_alpha_gomoku_quicktest.py`
基本的なクイックテスト版

**特徴:**
- 盤面サイズ: 6x6
- イテレーション: 3回
- ゲーム数: 4ゲーム/イテレーション
- 学習率: 0.01

**使用方法:**
```bash
python train_alpha_gomoku_quicktest.py
```

### 2. `train_alpha_gomoku_ultralight.py`
超軽量版（最小限の動作確認）

**特徴:**
- 盤面サイズ: 5x5
- イテレーション: 2回
- ゲーム数: 2ゲーム/イテレーション
- 基本動作テスト付き

**使用方法:**
```bash
python train_alpha_gomoku_ultralight.py
```

### 3. `train_alpha_gomoku_testsuite.py`
テストレベル選択可能版

**特徴:**
- 3つのテストレベル（ultra/quick/normal）
- コマンドライン引数対応
- 詳細な統計情報表示

**使用方法:**
```bash
# 超軽量版
python train_alpha_gomoku_testsuite.py --level ultra

# 軽量版
python train_alpha_gomoku_testsuite.py --level quick

# 通常版
python train_alpha_gomoku_testsuite.py --level normal

# 詳細ログ付き
python train_alpha_gomoku_testsuite.py --level quick --verbose

# トレーニング後に自動で対戦テスト
python train_alpha_gomoku_testsuite.py --level quick --play
```

## 推奨使用順序

1. **初回動作確認**: `train_alpha_gomoku_ultralight.py`
2. **基本テスト**: `train_alpha_gomoku_quicktest.py`
3. **詳細テスト**: `train_alpha_gomoku_testsuite.py --level quick`
4. **本格テスト**: `train_alpha_gomoku.py`

## 各レベルの比較

| レベル | 盤面サイズ | イテレーション | ゲーム数 | 予想時間 |
|--------|-----------|----------------|----------|----------|
| Ultra  | 5x5       | 1              | 2        | 30秒以内 |
| Quick  | 6x6       | 3              | 4        | 2-3分    |
| Normal | 8x8       | 5              | 8        | 5-10分   |

## トラブルシューティング

### よくあるエラー
1. **モジュールが見つからない**: パスの問題
2. **CUDA関連エラー**: GPU/CPU設定の問題
3. **メモリ不足**: 設定値が大きすぎる

### 解決方法
1. 最初は必ず`ultralight`版から実行
2. エラーが発生した場合は`--verbose`オプションを使用
3. GPUが使用できない場合は自動的にCPUに切り替わります

## 注意事項

- テスト版は動作確認を目的としており、実際のAI性能は限定的です
- 実際の訓練には元の`train_alpha_gomoku.py`または適切なパラメータを使用してください
- テスト用ディレクトリ（`models_test`、`logs_test`など）は定期的に削除してください
