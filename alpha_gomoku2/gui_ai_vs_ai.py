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

class AIvsAIGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("AlphaGomoku AI vs AI")
        self.master.geometry("1200x800")
        
        # ゲーム設定
        self.board_size = 8  # ボードサイズ
        self.cell_size = 40  # セルサイズ
        self.margin = 50     # マージン
        
        # ゲーム状態
        self.env = None
        self.ai_player1 = None  # AI1 (通常は先手)
        self.ai_player2 = None  # AI2 (通常は後手)
        self.game_over = False
        self.game_running = False
        self.auto_continue = False
        
        # 統計
        self.ai1_wins = 0
        self.ai2_wins = 0
        self.draws = 0
        self.total_games = 0
        
        # 詳細統計（先手後手別）
        self.ai1_wins_as_first = 0
        self.ai1_wins_as_second = 0
        self.ai2_wins_as_first = 0
        self.ai2_wins_as_second = 0
        
        # ゲーム速度
        self.game_speed = 1.0
        
        # GUI要素の初期化
        self.setup_gui()
        
        # AIモデルの読み込み
        self.load_ai_models()
        
    def setup_gui(self):
        """GUI要素をセットアップする"""
        # フレーム作成
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # 左側フレーム（ボード）
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side='left', expand=True, fill='both')
        
        # 右側フレーム（コントロール）
        self.right_frame = tk.Frame(self.main_frame, width=300)
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
            text="AI vs AI", 
            font=('Arial', 18, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # ゲーム情報
        self.info_frame = tk.Frame(self.right_frame)
        self.info_frame.pack(fill='x', pady=(0, 20))
        
        self.turn_label = tk.Label(
            self.info_frame, 
            text="準備中...", 
            font=('Arial', 12, 'bold')
        )
        self.turn_label.pack()
        
        self.status_label = tk.Label(
            self.info_frame, 
            text="AIモデル読み込み中...", 
            font=('Arial', 10)
        )
        self.status_label.pack()
        
        # AI1設定
        ai1_frame = tk.LabelFrame(self.right_frame, text="AI1 設定 (黒石)", font=('Arial', 12, 'bold'))
        ai1_frame.pack(fill='x', pady=(0, 10))
        
        self.ai1_playout_var = tk.IntVar(value=1000)
        ai1_scale = tk.Scale(
            ai1_frame, 
            from_=100, 
            to=10000, 
            resolution=100,
            orient='horizontal',
            variable=self.ai1_playout_var,
            label="MCTS シミュレーション回数"
        )
        ai1_scale.pack(fill='x', padx=5, pady=5)
        
        # AI1直接入力
        ai1_entry_frame = tk.Frame(ai1_frame)
        ai1_entry_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(ai1_entry_frame, text="直接入力:", font=('Arial', 10)).pack(side='left')
        
        self.ai1_entry = tk.Entry(ai1_entry_frame, width=10, font=('Arial', 10))
        self.ai1_entry.pack(side='left', padx=(5, 5))
        self.ai1_entry.insert(0, "1000")
        
        def update_ai1_from_entry():
            try:
                value = int(self.ai1_entry.get())
                if value > 0:
                    self.ai1_playout_var.set(value)
                    print(f"AI1プレイアウト回数を直接入力で更新: {value}")
                    if value > 10000:
                        ai1_scale.config(to=value + 5000)
            except ValueError:
                print("AI1: 無効な値が入力されました")
        
        def update_ai1_entry_from_scale(*args):
            new_value = self.ai1_playout_var.get()
            self.ai1_entry.delete(0, tk.END)
            self.ai1_entry.insert(0, str(new_value))
        
        tk.Button(ai1_entry_frame, text="適用", command=update_ai1_from_entry, font=('Arial', 9)).pack(side='left')
        self.ai1_playout_var.trace('w', update_ai1_entry_from_scale)
        
        # AI2設定
        ai2_frame = tk.LabelFrame(self.right_frame, text="AI2 設定 (白石)", font=('Arial', 12, 'bold'))
        ai2_frame.pack(fill='x', pady=(0, 20))
        
        self.ai2_playout_var = tk.IntVar(value=1000)
        ai2_scale = tk.Scale(
            ai2_frame, 
            from_=100, 
            to=10000, 
            resolution=100,
            orient='horizontal',
            variable=self.ai2_playout_var,
            label="MCTS シミュレーション回数"
        )
        ai2_scale.pack(fill='x', padx=5, pady=5)
        
        # AI2直接入力
        ai2_entry_frame = tk.Frame(ai2_frame)
        ai2_entry_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(ai2_entry_frame, text="直接入力:", font=('Arial', 10)).pack(side='left')
        
        self.ai2_entry = tk.Entry(ai2_entry_frame, width=10, font=('Arial', 10))
        self.ai2_entry.pack(side='left', padx=(5, 5))
        self.ai2_entry.insert(0, "1000")
        
        def update_ai2_from_entry():
            try:
                value = int(self.ai2_entry.get())
                if value > 0:
                    self.ai2_playout_var.set(value)
                    print(f"AI2プレイアウト回数を直接入力で更新: {value}")
                    if value > 10000:
                        ai2_scale.config(to=value + 5000)
            except ValueError:
                print("AI2: 無効な値が入力されました")
        
        def update_ai2_entry_from_scale(*args):
            new_value = self.ai2_playout_var.get()
            self.ai2_entry.delete(0, tk.END)
            self.ai2_entry.insert(0, str(new_value))
        
        tk.Button(ai2_entry_frame, text="適用", command=update_ai2_from_entry, font=('Arial', 9)).pack(side='left')
        self.ai2_playout_var.trace('w', update_ai2_entry_from_scale)
        
        # ゲーム速度設定
        speed_frame = tk.Frame(self.right_frame)
        speed_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(speed_frame, text="ゲーム速度:", font=('Arial', 12, 'bold')).pack()
        
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = tk.Scale(
            speed_frame, 
            from_=0.1, 
            to=5.0, 
            resolution=0.1,
            orient='horizontal',
            variable=self.speed_var,
            label="手番間隔 (秒)"
        )
        speed_scale.pack(fill='x')
        
        # 先手プレイヤー設定
        player_frame = tk.Frame(self.right_frame)
        player_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(player_frame, text="先手プレイヤー:", font=('Arial', 12, 'bold')).pack()
        
        self.first_player_var = tk.StringVar(value="alternating")
        tk.Radiobutton(
            player_frame, 
            text="AI1固定", 
            variable=self.first_player_var, 
            value="ai1"
        ).pack()
        tk.Radiobutton(
            player_frame, 
            text="AI2固定", 
            variable=self.first_player_var, 
            value="ai2"
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
            text="単発ゲーム開始", 
            command=self.start_single_game,
            font=('Arial', 12),
            bg='lightgreen'
        )
        self.start_button.pack(fill='x', pady=2)
        
        self.auto_button = tk.Button(
            button_frame, 
            text="連続ゲーム開始", 
            command=self.start_auto_games,
            font=('Arial', 12),
            bg='lightblue'
        )
        self.auto_button.pack(fill='x', pady=2)
        
        self.stop_button = tk.Button(
            button_frame, 
            text="ゲーム停止", 
            command=self.stop_games,
            font=('Arial', 12),
            bg='lightcoral',
            state='disabled'
        )
        self.stop_button.pack(fill='x', pady=2)
        
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
        
        tk.Label(stats_frame, text="統計:", font=('Arial', 14, 'bold')).pack()
        
        self.total_games_label = tk.Label(stats_frame, text="総ゲーム数: 0", font=('Arial', 11))
        self.total_games_label.pack()
        
        self.ai1_wins_label = tk.Label(stats_frame, text="AI1勝利: 0 (0.0%)", font=('Arial', 11))
        self.ai1_wins_label.pack()
        
        self.ai2_wins_label = tk.Label(stats_frame, text="AI2勝利: 0 (0.0%)", font=('Arial', 11))
        self.ai2_wins_label.pack()
        
        self.draws_label = tk.Label(stats_frame, text="引き分け: 0 (0.0%)", font=('Arial', 11))
        self.draws_label.pack()
        
        # 詳細統計（先手後手別）
        detail_frame = tk.Frame(self.right_frame)
        detail_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(detail_frame, text="詳細統計:", font=('Arial', 12, 'bold')).pack()
        
        self.ai1_first_label = tk.Label(detail_frame, text="AI1先手: 0勝", font=('Arial', 10))
        self.ai1_first_label.pack()
        
        self.ai1_second_label = tk.Label(detail_frame, text="AI1後手: 0勝", font=('Arial', 10))
        self.ai1_second_label.pack()
        
        self.ai2_first_label = tk.Label(detail_frame, text="AI2先手: 0勝", font=('Arial', 10))
        self.ai2_first_label.pack()
        
        self.ai2_second_label = tk.Label(detail_frame, text="AI2後手: 0勝", font=('Arial', 10))
        self.ai2_second_label.pack()
        
        # ゲーム進行情報
        progress_frame = tk.Frame(self.right_frame)
        progress_frame.pack(fill='x')
        
        tk.Label(progress_frame, text="進行状況:", font=('Arial', 12, 'bold')).pack()
        
        self.game_count_label = tk.Label(progress_frame, text="現在のゲーム: 0", font=('Arial', 10))
        self.game_count_label.pack()
        
        self.move_count_label = tk.Label(progress_frame, text="手数: 0", font=('Arial', 10))
        self.move_count_label.pack()
    
    def load_ai_models(self):
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
            
            # AI1プレイヤーの初期化
            initial_playout1 = self.ai1_playout_var.get()
            self.ai_player1 = MCTSPlayer(
                self.policy_value_net.policy_value_fn,
                c_puct=5,
                n_playout=initial_playout1,
                is_selfplay=0
            )
            
            # AI2プレイヤーの初期化
            initial_playout2 = self.ai2_playout_var.get()
            self.ai_player2 = MCTSPlayer(
                self.policy_value_net.policy_value_fn,
                c_puct=5,
                n_playout=initial_playout2,
                is_selfplay=0
            )
            
            print(f"AI1プレイヤー初期化完了: 初期プレイアウト回数 = {initial_playout1}")
            print(f"AI2プレイヤー初期化完了: 初期プレイアウト回数 = {initial_playout2}")
            
            print("AIモデル読み込み完了")
            self.status_label.config(text="モデル読み込み完了")
            self.start_button.config(state='normal')
            self.auto_button.config(state='normal')
            
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
        
        # 最後に置かれた石をハイライト
        if hasattr(self, 'last_move') and self.last_move is not None:
            x, y = self.last_move
            canvas_x = self.margin + x * self.cell_size
            canvas_y = self.margin + y * self.cell_size
            radius = self.cell_size // 4
            self.canvas.create_oval(
                canvas_x - radius, canvas_y - radius,
                canvas_x + radius, canvas_y + radius,
                outline='red', width=2, fill=''
            )
    
    def determine_first_player(self):
        """先手プレイヤーを決定する"""
        setting = self.first_player_var.get()
        if setting == "alternating":
            # 交互に切り替え
            return "ai1" if self.total_games % 2 == 0 else "ai2"
        else:
            # 固定設定
            return setting
    
    def start_single_game(self):
        """単発ゲームを開始する"""
        if not self.ai_player1 or not self.ai_player2:
            messagebox.showerror("エラー", "AIモデルが読み込まれていません")
            return
        
        self.auto_continue = False
        self.start_games()
    
    def start_auto_games(self):
        """連続ゲームを開始する"""
        if not self.ai_player1 or not self.ai_player2:
            messagebox.showerror("エラー", "AIモデルが読み込まれていません")
            return
        
        self.auto_continue = True
        self.start_games()
    
    def start_games(self):
        """ゲームを開始する"""
        self.game_running = True
        self.game_over = False
        self.last_move = None
        
        # ボタンの状態更新
        self.start_button.config(state='disabled')
        self.auto_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # ゲームスレッドを開始
        self.game_thread = threading.Thread(target=self.run_game_loop, daemon=True)
        self.game_thread.start()
    
    def stop_games(self):
        """ゲームを停止する"""
        self.game_running = False
        self.auto_continue = False
        
        # ボタンの状態更新
        self.start_button.config(state='normal')
        self.auto_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.status_label.config(text="ゲーム停止")
        self.turn_label.config(text="停止中...")
    
    def reset_stats(self):
        """統計をリセットする"""
        self.ai1_wins = 0
        self.ai2_wins = 0
        self.draws = 0
        self.total_games = 0
        
        # 詳細統計もリセット
        self.ai1_wins_as_first = 0
        self.ai1_wins_as_second = 0
        self.ai2_wins_as_first = 0
        self.ai2_wins_as_second = 0
        
        self.update_stats()
        self.game_count_label.config(text="現在のゲーム: 0")
        self.move_count_label.config(text="手数: 0")
    
    def run_game_loop(self):
        """ゲームループを実行する（別スレッド）"""
        while self.game_running:
            try:
                self.play_single_game()
                if self.game_running and not self.auto_continue:
                    # 単発ゲームの場合は終了
                    break
                elif self.game_running and self.auto_continue:
                    # 連続ゲームの場合は短い間隔で継続
                    time.sleep(0.5)
            except Exception as e:
                print(f"ゲーム実行エラー: {str(e)}")
                traceback.print_exc()
                self.master.after(0, lambda: self.status_label.config(text="ゲームエラー"))
                break
        
        # ゲーム終了処理
        self.master.after(0, self.end_games)
    
    def play_single_game(self):
        """一回のゲームを実行する"""
        if not self.game_running:
            return
        
        # 環境の初期化
        self.env = GomokuEnv(board_size=self.board_size)
        self.last_move = None
        
        # 先手プレイヤーの決定
        first_player = self.determine_first_player()
        ai1_is_first = (first_player == "ai1")
        
        # 現在のゲーム番号を更新
        current_game = self.total_games + 1
        self.master.after(0, lambda: self.game_count_label.config(text=f"現在のゲーム: {current_game}"))
        
        # ゲーム開始メッセージ
        self.master.after(0, lambda: self.turn_label.config(
            text=f"ゲーム{current_game} (先手: {'AI1' if ai1_is_first else 'AI2'})"
        ))
        self.master.after(0, self.draw_board)
        
        move_count = 0
        max_moves = self.board_size * self.board_size
        
        while not self.game_over and self.game_running and move_count < max_moves:
            # 現在のプレイヤーがAI1かAI2かを判定
            current_player_is_ai1 = (ai1_is_first and self.env.current_player == 1) or \
                                   (not ai1_is_first and self.env.current_player == 2)
            
            if current_player_is_ai1:
                # AI1の手番
                current_ai = self.ai_player1
                ai_name = "AI1"
                playout_count = self.ai1_playout_var.get()
                current_ai.n_playout = playout_count
            else:
                # AI2の手番
                current_ai = self.ai_player2
                ai_name = "AI2"
                playout_count = self.ai2_playout_var.get()
                current_ai.n_playout = playout_count
            
            # 手番表示
            self.master.after(0, lambda name=ai_name: self.turn_label.config(text=f"{name}の手番"))
            self.master.after(0, lambda name=ai_name, count=playout_count: 
                            self.status_label.config(text=f"{name}思考中 (プレイアウト: {count})"))
            
            # AIの手を取得
            action = current_ai.get_action(self.env)
            
            if action is None:
                # 有効な手がない場合は引き分け
                self.game_over = True
                self.draws += 1
                result = "引き分け"
                break
            
            # 手を実行
            _, reward, done, _ = self.env.step(action)
            move_count += 1
            
            # 最後の手を記録
            self.last_move = action
            
            # 盤面更新
            self.master.after(0, self.draw_board)
            
            # 手数更新
            self.master.after(0, lambda count=move_count: 
                            self.move_count_label.config(text=f"手数: {count}"))
            
            print(f"{ai_name}の手: {action}")
            
            if done:
                # 勝者の判定
                winner = 3 - self.env.current_player  # 前のプレイヤーが勝者
                if (ai1_is_first and winner == 1) or (not ai1_is_first and winner == 2):
                    # AI1の勝利
                    self.ai1_wins += 1
                    if ai1_is_first:
                        self.ai1_wins_as_first += 1
                    else:
                        self.ai1_wins_as_second += 1
                    result = "AI1の勝利"
                else:
                    # AI2の勝利
                    self.ai2_wins += 1
                    if not ai1_is_first:
                        self.ai2_wins_as_first += 1
                    else:
                        self.ai2_wins_as_second += 1
                    result = "AI2の勝利"
                
                self.game_over = True
                break
            
            # 手番間の待機
            if self.game_running:
                time.sleep(self.speed_var.get())
        
        # ゲーム終了処理
        if self.game_running:
            if move_count >= max_moves and not self.game_over:
                self.draws += 1
                result = "引き分け（最大手数）"
            
            self.total_games += 1
            
            self.master.after(0, lambda: self.turn_label.config(text=f"ゲーム終了: {result}"))
            self.master.after(0, lambda: self.status_label.config(text=f"結果: {result}"))
            self.master.after(0, self.update_stats)
            
            print(f"ゲーム{self.total_games}終了: {result}")
    
    def end_games(self):
        """ゲーム終了処理"""
        self.game_running = False
        self.auto_continue = False
        
        # ボタンの状態更新
        self.start_button.config(state='normal')
        self.auto_button.config(state='normal')
        self.stop_button.config(state='disabled')
    
    def update_stats(self):
        """統計表示を更新する"""
        self.total_games_label.config(text=f"総ゲーム数: {self.total_games}")
        
        if self.total_games > 0:
            ai1_percentage = (self.ai1_wins / self.total_games) * 100
            ai2_percentage = (self.ai2_wins / self.total_games) * 100
            draw_percentage = (self.draws / self.total_games) * 100
            
            self.ai1_wins_label.config(text=f"AI1勝利: {self.ai1_wins} ({ai1_percentage:.1f}%)")
            self.ai2_wins_label.config(text=f"AI2勝利: {self.ai2_wins} ({ai2_percentage:.1f}%)")
            self.draws_label.config(text=f"引き分け: {self.draws} ({draw_percentage:.1f}%)")
            
            # 詳細統計も更新
            self.ai1_first_label.config(text=f"AI1先手: {self.ai1_wins_as_first}勝")
            self.ai1_second_label.config(text=f"AI1後手: {self.ai1_wins_as_second}勝")
            self.ai2_first_label.config(text=f"AI2先手: {self.ai2_wins_as_first}勝")
            self.ai2_second_label.config(text=f"AI2後手: {self.ai2_wins_as_second}勝")
        else:
            self.ai1_wins_label.config(text="AI1勝利: 0 (0.0%)")
            self.ai2_wins_label.config(text="AI2勝利: 0 (0.0%)")
            self.draws_label.config(text="引き分け: 0 (0.0%)")
            
            # 詳細統計も初期化
            self.ai1_first_label.config(text="AI1先手: 0勝")
            self.ai1_second_label.config(text="AI1後手: 0勝")
            self.ai2_first_label.config(text="AI2先手: 0勝")
            self.ai2_second_label.config(text="AI2後手: 0勝")

def main():
    root = tk.Tk()
    app = AIvsAIGUI(root)
    
    def on_closing():
        app.game_running = False
        if hasattr(app, 'game_thread') and app.game_thread.is_alive():
            app.game_thread.join(timeout=1.0)
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
