@echo off
echo AlphaZero五目並べ - 対戦実行スクリプト
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

REM 訓練済みモデルの確認
if not exist "models" (
    echo 警告: modelsディレクトリが見つかりません。
    echo 先に訓練を実行してください。
    echo.
    set /p train_first="今すぐ訓練を実行しますか？ (y/n): "
    if /i "%train_first%"=="y" (
        call run_training.bat
    ) else (
        pause
        exit /b 1
    )
)

echo.
echo 盤面サイズを選択してください:
echo 1. 8x8 (標準)
echo 2. 15x15 (本格的)
echo 3. カスタム
echo.
set /p choice="選択 (1-3): "

if "%choice%"=="1" (
    echo 8x8盤面で対戦を開始します...
    python run_alpha_gomoku.py --play --board-size 8
) else if "%choice%"=="2" (
    echo 15x15盤面で対戦を開始します...
    python run_alpha_gomoku.py --play --board-size 15
) else if "%choice%"=="3" (
    echo.
    set /p board_size="ボードサイズを入力 (例: 10): "
    if "%board_size%"=="" set board_size=8
    echo %board_size%x%board_size%盤面で対戦を開始します...
    python run_alpha_gomoku.py --play --board-size %board_size%
) else (
    echo 無効な選択です。
    pause
    exit /b 1
)

echo.
echo 対戦が終了しました。
pause
