import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import torch
from GomokuEnv import GomokuEnv
from network import PolicyValueNet
from mcts import MCTSPlayer
import threading
import time
import traceback
import sys
import random

class RandomPlayer:
    """ランダムに手を選ぶプレイヤー"""
    def __init__(self, board_size):
        self.board_size = board_size
    
    def get_action(self, board):
        """有効な手からランダムに選択"""
        valid_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board[y][x] == 0:  # 空きマス
                    valid_moves.append((x, y))
        
        if not valid_moves:
            return None  # 有効な手がない
        
        return random.choice(valid_moves)

class RandomVsAIGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("ランダムプレイヤー vs AlphaGomoku AI")
        self.master.geometry("1000x700")
        
        # ゲーム設定
        self.board_size = 6  # ボードサイズ
        self.cell_size = 40  # セルサイズ
        self.margin = 50     # マージン
        
        # ゲーム状態
        self.env = None
        self.ai_player = None
        self.random_player = None
        self.game_over = False
        self.game_running = False
        
        # 統計
        self.ai_wins = 0
        self.random_wins = 0
        self.draws = 0
        self.total_games = 0
        self.current_first_player = "random"  # 現在のゲームの先手プレイヤー
        
        # 詳細統計（先手後手別）
        self.ai_wins_as_first = 0
        self.ai_wins_as_second = 0
        self.random_wins_as_first = 0
        self.random_wins_as_second = 0
        
        # GUI要素の初期化
        self.setup_gui()
        
        # AIモデルの読み込み
        self.load_ai_model()
        
    def setup_gui(self):
        """GUI要素をセットアップする"""
        # フレーム作成
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # 左側フレーム（ボード）
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side='left', expand=True, fill='both')
        
        # 右側フレーム（コントロール）
        self.right_frame = tk.Frame(self.main_frame, width=250)
        self.right_frame.pack(side='right', fill='y', padx=(10, 0))
        self.right_frame.pack_propagate(False)
        
        # キャンバス（ボード描画用）
        canvas_size = self.board_size * self.cell_size + 2 * self.margin
        self.canvas = tk.Canvas(
            self.left_frame, 
            width=canvas_size, 
            height=canvas_size, 
            bg='burlywood'
        )
        self.canvas.pack(expand=True)
        
        # コントロールパネル
        self.setup_controls()
        
        # ボードの描画
        self.draw_board()
    
    def setup_controls(self):
        """コントロールパネルをセットアップする"""
        # タイトル
        title_label = tk.Label(
            self.right_frame, 
            text="ランダム vs AI", 
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # ゲーム情報
        self.info_frame = tk.Frame(self.right_frame)
        self.info_frame.pack(fill='x', pady=(0, 20))
        
        self.turn_label = tk.Label(
            self.info_frame, 
            text="準備中...", 
            font=('Arial', 12)
        )
        self.turn_label.pack()
        
        self.status_label = tk.Label(
            self.info_frame, 
            text="AIモデル読み込み中...", 
            font=('Arial', 10)
        )
        self.status_label.pack()
        
        # ゲーム速度設定（削除）
        # MCTSが完了するまでしっかり待機するため、時間制限を削除
        
        # AI思考時間設定
        ai_frame = tk.Frame(self.right_frame)
        ai_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(ai_frame, text="AI思考時間:", font=('Arial', 12, 'bold')).pack()
        
        self.ai_playout_var = tk.IntVar(value=400)
        ai_scale = tk.Scale(
            ai_frame, 
            from_=100, 
            to=50000, 
            resolution=100,
            orient='horizontal',
            variable=self.ai_playout_var,
            label="MCTS シミュレーション回数"
        )
        ai_scale.pack(fill='x')
        
        # 直接入力用のエントリー
        entry_frame = tk.Frame(ai_frame)
        entry_frame.pack(fill='x', pady=(5, 0))
        
        tk.Label(entry_frame, text="直接入力:", font=('Arial', 10)).pack(side='left')
        
        self.ai_entry = tk.Entry(entry_frame, width=10, font=('Arial', 10))
        self.ai_entry.pack(side='left', padx=(5, 5))
        self.ai_entry.insert(0, "1000")
        
        def update_from_entry():
            try:
                value = int(self.ai_entry.get())
                if value > 0:
                    self.ai_playout_var.set(value)
                    print(f"プレイアウト回数を直接入力で更新: {value}")
                    # スケールの範囲を動的に調整
                    if value > 50000:
                        ai_scale.config(to=value + 10000)
                else:
                    print("プレイアウト回数は正の値である必要があります")
            except ValueError:
                print("無効な値が入力されました")
                pass
        
        def update_entry_from_scale(*args):
            new_value = self.ai_playout_var.get()
            self.ai_entry.delete(0, tk.END)
            self.ai_entry.insert(0, str(new_value))
            print(f"プレイアウト回数をスケールで更新: {new_value}")
        
        tk.Button(entry_frame, text="適用", command=update_from_entry, font=('Arial', 9)).pack(side='left')
        
        # スケールが変更されたときにエントリーも更新
        self.ai_playout_var.trace('w', update_entry_from_scale)
        
        # プレイヤー設定
        player_frame = tk.Frame(self.right_frame)
        player_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(player_frame, text="先手プレイヤー:", font=('Arial', 12, 'bold')).pack()
        
        self.first_player_var = tk.StringVar(value="alternating")
        tk.Radiobutton(
            player_frame, 
            text="ランダムプレイヤー固定", 
            variable=self.first_player_var, 
            value="random"
        ).pack()
        tk.Radiobutton(
            player_frame, 
            text="AIプレイヤー固定", 
            variable=self.first_player_var, 
            value="ai"
        ).pack()
        tk.Radiobutton(
            player_frame, 
            text="交互に切り替え", 
            variable=self.first_player_var, 
            value="alternating"
        ).pack()
        
        # コントロールボタン
        button_frame = tk.Frame(self.right_frame)
        button_frame.pack(fill='x', pady=(0, 20))
        
        self.start_button = tk.Button(
            button_frame, 
            text="ゲーム開始", 
            command=self.start_game,
            font=('Arial', 12),
            bg='lightgreen'
        )
        self.start_button.pack(fill='x', pady=2)
        
        self.stop_button = tk.Button(
            button_frame, 
            text="ゲーム停止", 
            command=self.stop_game,
            font=('Arial', 12),
            bg='lightcoral',
            state='disabled'
        )
        self.stop_button.pack(fill='x', pady=2)
        
        # 連続ゲーム設定
        continuous_frame = tk.Frame(button_frame)
        continuous_frame.pack(fill='x', pady=2)
        
        self.continuous_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            continuous_frame,
            text="連続ゲーム",
            variable=self.continuous_var,
            font=('Arial', 10)
        ).pack()
        
        self.reset_stats_button = tk.Button(
            button_frame, 
            text="統計リセット", 
            command=self.reset_stats,
            font=('Arial', 12)
        )
        self.reset_stats_button.pack(fill='x', pady=2)
        
        # 統計表示
        stats_frame = tk.Frame(self.right_frame)
        stats_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(stats_frame, text="統計:", font=('Arial', 12, 'bold')).pack()
        
        self.total_games_label = tk.Label(stats_frame, text="総ゲーム数: 0", font=('Arial', 10))
        self.total_games_label.pack()
        
        self.ai_wins_label = tk.Label(stats_frame, text="AI勝利: 0 (0.0%)", font=('Arial', 10))
        self.ai_wins_label.pack()
        
        self.random_wins_label = tk.Label(stats_frame, text="ランダム勝利: 0 (0.0%)", font=('Arial', 10))
        self.random_wins_label.pack()
        
        self.draws_label = tk.Label(stats_frame, text="引き分け: 0 (0.0%)", font=('Arial', 10))
        self.draws_label.pack()
        
        # 詳細統計（先手後手別）
        detail_frame = tk.Frame(self.right_frame)
        detail_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(detail_frame, text="詳細統計:", font=('Arial', 10, 'bold')).pack()
        
        self.ai_first_label = tk.Label(detail_frame, text="AI先手: 0勝", font=('Arial', 9))
        self.ai_first_label.pack()
        
        self.ai_second_label = tk.Label(detail_frame, text="AI後手: 0勝", font=('Arial', 9))
        self.ai_second_label.pack()
        
        self.random_first_label = tk.Label(detail_frame, text="ランダム先手: 0勝", font=('Arial', 9))
        self.random_first_label.pack()
        
        self.random_second_label = tk.Label(detail_frame, text="ランダム後手: 0勝", font=('Arial', 9))
        self.random_second_label.pack()
    
    def load_ai_model(self):
        """AIモデルを読み込む"""
        try:
            print("AIモデル読み込み開始...")
            self.status_label.config(text="AIモデル読み込み中...")
            
            # 一時的な環境を作成（PolicyValueNetの初期化用）
            temp_env = GomokuEnv(board_size=self.board_size)
            
            # PolicyValueNetの初期化
            use_gpu = torch.cuda.is_available()
            print(f"GPU使用可能: {use_gpu}")
            
            self.policy_value_net = PolicyValueNet(
                board_size=self.board_size,
                model_file="current_policy.model",
                use_gpu=use_gpu,
                env=temp_env
            )
            
            # MCTSプレイヤーの初期化（初期値として1000を使用）
            initial_playout = self.ai_playout_var.get()
            self.ai_player = MCTSPlayer(
                self.policy_value_net.policy_value_fn,
                c_puct=5,
                n_playout=initial_playout,  # 初期値、後で動的に変更
                is_selfplay=0
            )
            
            print(f"MCTSプレイヤー初期化完了: 初期プレイアウト回数 = {initial_playout}")
            
            # ランダムプレイヤーの初期化
            self.random_player = RandomPlayer(self.board_size)
            
            print("AIモデル読み込み完了")
            self.status_label.config(text="モデル読み込み完了")
            self.start_button.config(state='normal')
            
        except Exception as e:
            error_msg = f"AIモデルの読み込みに失敗しました: {str(e)}"
            print(f"エラー: {error_msg}")
            print("詳細なエラー情報:")
            traceback.print_exc()
            self.status_label.config(text="モデル読み込み失敗")
            messagebox.showerror("エラー", error_msg)
    
    def draw_board(self):
        """ボードを描画する"""
        self.canvas.delete("all")
        
        # グリッドライン
        for i in range(self.board_size):
            # 縦線
            x = self.margin + i * self.cell_size
            self.canvas.create_line(
                x, self.margin, 
                x, self.margin + (self.board_size - 1) * self.cell_size,
                fill='black', width=1
            )
            # 横線
            y = self.margin + i * self.cell_size
            self.canvas.create_line(
                self.margin, y,
                self.margin + (self.board_size - 1) * self.cell_size, y,
                fill='black', width=1
            )
        
        # 石を描画
        if hasattr(self, 'env') and self.env is not None:
            board = self.env.board.GetBoardInt()
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if board[row][col] != 0:
                        x = self.margin + col * self.cell_size
                        y = self.margin + row * self.cell_size
                        radius = self.cell_size // 3
                        
                        if board[row][col] == 1:  # 黒石
                            self.canvas.create_oval(
                                x - radius, y - radius,
                                x + radius, y + radius,
                                fill='black', outline='black'
                            )
                        else:  # 白石
                            self.canvas.create_oval(
                                x - radius, y - radius,
                                x + radius, y + radius,
                                fill='white', outline='black'
                            )
    
    def start_game(self):
        """ゲームを開始する"""
        if not self.ai_player or not self.random_player:
            messagebox.showerror("エラー", "AIモデルが読み込まれていません")
            return
        
        self.game_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # 新しいスレッドでゲームを実行
        self.game_thread = threading.Thread(target=self.run_game_loop, daemon=True)
        self.game_thread.start()
    
    def stop_game(self):
        """ゲームを停止する"""
        self.game_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="ゲーム停止")
        self.turn_label.config(text="停止中...")
    
    def determine_first_player(self):
        """先手プレイヤーを決定する"""
        setting = self.first_player_var.get()
        if setting == "alternating":
            # 交互に切り替え
            if self.total_games % 2 == 0:
                return "random"
            else:
                return "ai"
        else:
            # 固定設定
            return setting
    
    def reset_stats(self):
        """統計をリセットする"""
        self.ai_wins = 0
        self.random_wins = 0
        self.draws = 0
        self.total_games = 0
        self.current_first_player = "random"  # リセット時は初期値に戻す
        
        # 詳細統計もリセット
        self.ai_wins_as_first = 0
        self.ai_wins_as_second = 0
        self.random_wins_as_first = 0
        self.random_wins_as_second = 0
        
        self.update_stats()
    
    def run_game_loop(self):
        """ゲームループを実行する（別スレッド）"""
        if self.continuous_var.get():
            # 連続ゲームモード
            while self.game_running:
                try:
                    self.play_single_game()
                    if self.game_running:
                        time.sleep(0.5)  # ゲーム間の短い間隔のみ残す
                except Exception as e:
                    print(f"ゲーム実行エラー: {str(e)}")
                    traceback.print_exc()
                    self.master.after(0, lambda: self.status_label.config(text="ゲームエラー"))
                    break
        else:
            # 単発ゲームモード
            try:
                self.play_single_game()
            except Exception as e:
                print(f"ゲーム実行エラー: {str(e)}")
                traceback.print_exc()
                self.master.after(0, lambda: self.status_label.config(text="ゲームエラー"))
            finally:
                self.master.after(0, self.stop_game)
    
    def play_single_game(self):
        """一回のゲームを実行する"""
        if not self.game_running:
            return
        
        # 環境の初期化
        self.env = GomokuEnv(board_size=self.board_size)
        
        # プレイヤーの決定（設定に応じて決定）
        first_player_type = self.determine_first_player()
        ai_is_first = (first_player_type == "ai")
        
        # 現在の先手プレイヤーを記録
        self.current_first_player = first_player_type
        
        self.master.after(0, lambda: self.turn_label.config(
            text=f"ゲーム{self.total_games + 1} (先手: {'AI' if ai_is_first else 'ランダム'})"
        ))
        self.master.after(0, self.draw_board)
        
        game_over = False
        move_count = 0
        max_moves = self.board_size * self.board_size
        
        while not game_over and self.game_running and move_count < max_moves:
            current_player_is_ai = (ai_is_first and self.env.current_player == 1) or \
                                 (not ai_is_first and self.env.current_player == 2)
            
            if current_player_is_ai:
                # AIの手番
                self.master.after(0, lambda: self.turn_label.config(text="AIの思考中..."))
                self.master.after(0, lambda: self.status_label.config(text="AI思考中"))
                
                # AI思考時間を動的に更新
                if hasattr(self, 'ai_player'):
                    playout_count = self.ai_playout_var.get()
                    self.ai_player.n_playout = playout_count
                    print(f"AI思考中: MCTSプレイアウト回数 = {playout_count}")
                    self.master.after(0, lambda: self.status_label.config(text=f"AI思考中 (プレイアウト: {playout_count})"))
                
                board = self.env.board.GetBoardInt()
                action = self.ai_player.get_action(self.env)
                
                player_name = "AI"
            else:
                # ランダムプレイヤーの手番
                self.master.after(0, lambda: self.turn_label.config(text="ランダムプレイヤーの手番"))
                self.master.after(0, lambda: self.status_label.config(text="ランダム思考中"))
                
                board = self.env.board.GetBoardInt()
                action = self.random_player.get_action(board)
                
                player_name = "ランダム"
            
            if action is None:
                # 有効な手がない場合は引き分け
                game_over = True
                self.draws += 1
                result = "引き分け"
                break
            
            # 手を実行
            _, reward, done, _ = self.env.step(action)
            move_count += 1
            
            # 盤面更新
            self.master.after(0, self.draw_board)
            
            if done:
                game_over = True
                # 勝者の判定
                winner = 3 - self.env.current_player  # 前のプレイヤーが勝者
                if (ai_is_first and winner == 1) or (not ai_is_first and winner == 2):
                    self.ai_wins += 1
                    if ai_is_first:
                        self.ai_wins_as_first += 1
                    else:
                        self.ai_wins_as_second += 1
                    result = "AI勝利"
                else:
                    self.random_wins += 1
                    if not ai_is_first:
                        self.random_wins_as_first += 1
                    else:
                        self.random_wins_as_second += 1
                    result = "ランダム勝利"
            
            # 手番の遅延を削除 - MCTSが完了するまで待機
            # if self.game_running:
            #     time.sleep(self.speed_var.get())
        
        # ゲーム終了処理
        if self.game_running:
            if move_count >= max_moves and not game_over:
                self.draws += 1
                result = "引き分け（最大手数）"
            
            self.total_games += 1
            
            self.master.after(0, lambda: self.turn_label.config(text=f"ゲーム終了: {result}"))
            self.master.after(0, lambda: self.status_label.config(text=f"結果: {result}"))
            self.master.after(0, self.update_stats)
    
    def update_stats(self):
        """統計表示を更新する"""
        self.total_games_label.config(text=f"総ゲーム数: {self.total_games}")
        
        if self.total_games > 0:
            ai_percentage = (self.ai_wins / self.total_games) * 100
            random_percentage = (self.random_wins / self.total_games) * 100
            draw_percentage = (self.draws / self.total_games) * 100
            
            self.ai_wins_label.config(text=f"AI勝利: {self.ai_wins} ({ai_percentage:.1f}%)")
            self.random_wins_label.config(text=f"ランダム勝利: {self.random_wins} ({random_percentage:.1f}%)")
            self.draws_label.config(text=f"引き分け: {self.draws} ({draw_percentage:.1f}%)")
            
            # 詳細統計も更新
            self.ai_first_label.config(text=f"AI先手: {self.ai_wins_as_first}勝")
            self.ai_second_label.config(text=f"AI後手: {self.ai_wins_as_second}勝")
            self.random_first_label.config(text=f"ランダム先手: {self.random_wins_as_first}勝")
            self.random_second_label.config(text=f"ランダム後手: {self.random_wins_as_second}勝")
        else:
            self.ai_wins_label.config(text="AI勝利: 0 (0.0%)")
            self.random_wins_label.config(text="ランダム勝利: 0 (0.0%)")
            self.draws_label.config(text="引き分け: 0 (0.0%)")
            
            # 詳細統計も初期化
            self.ai_first_label.config(text="AI先手: 0勝")
            self.ai_second_label.config(text="AI後手: 0勝")
            self.random_first_label.config(text="ランダム先手: 0勝")
            self.random_second_label.config(text="ランダム後手: 0勝")

def main():
    root = tk.Tk()
    app = RandomVsAIGUI(root)
    
    def on_closing():
        app.game_running = False
        if hasattr(app, 'game_thread') and app.game_thread.is_alive():
            app.game_thread.join(timeout=1.0)
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
