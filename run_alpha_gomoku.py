"""
AlphaZero五目並べの実行用スクリプト
使用方法:
1. 訓練実行: python run_alpha_gomoku.py --train
2. 人間との対戦: python run_alpha_gomoku.py --play
3. カスタム設定での訓練: python run_alpha_gomoku.py --train --board-size 15 --iterations 100
"""
import argparse
import sys
import os
from alpha_gomoku import AlphaZero, DEFAULT_BOARD_SIZE, DEFAULT_ITERATIONS, DEFAULT_SELF_PLAY_GAMES, DEFAULT_LEARNING_RATE

def main():
    parser = argparse.ArgumentParser(description='AlphaZero五目並べ - 実行用スクリプト')
    
    # 実行モード
    parser.add_argument('--train', action='store_true', help='AIを訓練する')
    parser.add_argument('--play', action='store_true', help='人間と対戦する')
    
    # パラメータ設定
    parser.add_argument('--board-size', type=int, default=DEFAULT_BOARD_SIZE,
                       help=f'盤面のサイズ (デフォルト: {DEFAULT_BOARD_SIZE}x{DEFAULT_BOARD_SIZE})')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS,
                       help=f'訓練イテレーション数 (デフォルト: {DEFAULT_ITERATIONS})')
    parser.add_argument('--self-play-games', type=int, default=DEFAULT_SELF_PLAY_GAMES,
                       help=f'イテレーションごとの自己対戦ゲーム数 (デフォルト: {DEFAULT_SELF_PLAY_GAMES})')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                       help=f'初期学習率 (デフォルト: {DEFAULT_LEARNING_RATE})')
    
    # ディレクトリ設定
    parser.add_argument('--model-dir', type=str, default='models',
                       help='モデル保存ディレクトリ (デフォルト: models)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='ログ保存ディレクトリ (デフォルト: logs)')
    
    # 高速設定（テスト用）
    parser.add_argument('--quick', action='store_true',
                       help='高速テスト設定（小さいパラメータで実行）')
    
    args = parser.parse_args()
    
    # 実行モードの確認
    if not args.train and not args.play:
        print("エラー: --train または --play のいずれかを指定してください")
        parser.print_help()
        sys.exit(1)
    
    if args.train and args.play:
        print("エラー: --train と --play は同時に指定できません")
        parser.print_help()
        sys.exit(1)
    
    # 高速設定の適用
    if args.quick:
        print("高速テスト設定を適用します...")
        args.board_size = min(args.board_size, 8)
        args.iterations = min(args.iterations, 5)
        args.self_play_games = min(args.self_play_games, 16)
        print(f"設定変更: ボードサイズ={args.board_size}, イテレーション={args.iterations}, ゲーム数={args.self_play_games}")
    
    # 設定の表示
    print("=" * 60)
    print("AlphaZero五目並べ - 設定確認")
    print("=" * 60)
    print(f"実行モード: {'訓練' if args.train else '対戦'}")
    print(f"ボードサイズ: {args.board_size}x{args.board_size}")
    if args.train:
        print(f"訓練イテレーション数: {args.iterations}")
        print(f"自己対戦ゲーム数/イテレーション: {args.self_play_games}")
        print(f"初期学習率: {args.lr}")
        print(f"モデル保存先: {args.model_dir}")
        print(f"ログ保存先: {args.log_dir}")
    print("=" * 60)
    
    try:
        # AlphaZeroの初期化
        alpha_zero = AlphaZero(
            board_size=args.board_size,
            num_iterations=args.iterations,
            num_self_play_games=args.self_play_games,
            checkpoint_dir=args.model_dir,
            log_dir=args.log_dir,
            initial_lr=args.lr
        )
        
        if args.train:
            print("訓練を開始します...")
            print("注意: 訓練には長時間かかる場合があります。Ctrl+Cで中断できます。")
            alpha_zero.train()
            print("訓練が完了しました！")
            
            # 訓練後に対戦するか確認
            play_after_train = input("\n訓練完了後に人間と対戦しますか？ (y/n): ").lower() == 'y'
            if play_after_train:
                print("\n人間との対戦を開始します...")
                alpha_zero.play_against_human()
        
        elif args.play:
            print("人間との対戦を開始します...")
            print("ゲームの説明:")
            print("- 座標を入力して石を置きます（例: 列3, 行4）")
            print("- 5つ並べると勝利です")
            print("- 盤面の座標は0から始まります")
            alpha_zero.play_against_human()
    
    except KeyboardInterrupt:
        print("\n\n実行が中断されました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
