@echo off
echo AlphaZero五目並べ - 訓練実行スクリプト
echo =====================================

REM 仮想環境の確認
if exist "venv\Scripts\activate.bat" (
    echo 仮想環境を有効化します...
    call venv\Scripts\activate.bat
) else (
    echo 注意: 仮想環境が見つかりません。グローバル環境で実行します。
)

REM Pythonの確認
python --version >nul 2>&1
if errorlevel 1 (
    echo エラー: Pythonが見つかりません。
    pause
    exit /b 1
)

REM 必要なパッケージの確認
echo 必要なパッケージを確認中...
python -c "import torch, numpy, matplotlib, tqdm, numba" >nul 2>&1
if errorlevel 1 (
    echo エラー: 必要なパッケージがインストールされていません。
    echo 以下のコマンドでインストールしてください:
    echo pip install torch numpy matplotlib tqdm numba
    pause
    exit /b 1
)

echo.
echo 訓練設定を選択してください:
echo 1. 標準設定（8x8盤面、長時間）
echo 2. 高速テスト（小さい設定、短時間）
echo 3. カスタム設定
echo.
set /p choice="選択 (1-3): "

if "%choice%"=="1" (
    echo 標準設定で訓練を開始します...
    python run_alpha_gomoku.py --train
) else if "%choice%"=="2" (
    echo 高速テスト設定で訓練を開始します...
    python run_alpha_gomoku.py --train --quick
) else if "%choice%"=="3" (
    echo.
    set /p board_size="ボードサイズ (デフォルト: 8): "
    set /p iterations="イテレーション数 (デフォルト: 10000): "
    set /p games="自己対戦ゲーム数 (デフォルト: 192): "
    
    if "%board_size%"=="" set board_size=8
    if "%iterations%"=="" set iterations=10000
    if "%games%"=="" set games=192
    
    echo カスタム設定で訓練を開始します...
    python run_alpha_gomoku.py --train --board-size %board_size% --iterations %iterations% --self-play-games %games%
) else (
    echo 無効な選択です。
    pause
    exit /b 1
)

echo.
echo 訓練が完了しました。
pause
