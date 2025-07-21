import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import torch
from GomokuEnv import GomokuEnv
import threading
import time
import traceback
import sys
import random
import copy

class PureMCTSPlayer:
    """学習無しのMCTSプレイヤー（ランダムプレイアウト）"""
    def __init__(self, board_size, c_puct=5, n_playout=200):
        self.board_size = board_size
        self.c_puct = c_puct
        self.n_playout = n_playout
        
    def get_action(self, env):
        """純粋なMCTSで行動を選択"""
        return self.mcts_pure(env)
    
    def mcts_pure(self, env):
        """純粋なMCTSアルゴリズム（より効率的な実装）"""
        # 有効な手を取得
        valid_moves = self.get_valid_moves(env)
        if not valid_moves:
            return None
        
        # 手が少ない場合は軽量化
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # プレイアウト回数を動的に調整
        playout_per_move = max(10, self.n_playout // len(valid_moves))
        
        # 各手の価値を計算
        move_values = {}
        for move in valid_moves:
            total_value = 0
            for _ in range(playout_per_move):
                # 環境をコピーして手を打つ
                env_copy = copy.deepcopy(env)
                _, _, done, info = env_copy.step(move)
                
                # ゲームが終了していない場合はランダムプレイアウト
                if not done:
                    value = self.random_playout(env_copy)
                else:
                    # ゲームが終了している場合は結果を評価
                    winner = info.get('win_player', 0)
                    if winner == 0:
                        value = 0  # 引き分け
                    else:
                        # 現在のプレイヤーが勝利したかどうか
                        value = 1.0 if winner == env.current_player else -1.0
                
                total_value += value
            
            move_values[move] = total_value / playout_per_move
        
        # 最も価値の高い手を選択
        best_move = max(move_values, key=move_values.get)
        return best_move
    
    def random_playout(self, env):
        """ランダムプレイアウトで最終的な価値を計算（最大手数制限付き）"""
        original_player = env.current_player
        max_playout_moves = 50  # 最大プレイアウト手数を制限
        move_count = 0
        
        while move_count < max_playout_moves:
            valid_moves = self.get_valid_moves(env)
            if not valid_moves:
                return 0  # 引き分け
            
            # ランダムに手を選択
            move = random.choice(valid_moves)
            _, _, done, info = env.step(move)
            move_count += 1
            
            if done:
                winner = info.get('win_player', 0)
                if winner == 0:
                    return 0  # 引き分け
                else:
                    # 元のプレイヤーが勝利したかどうか
                    return -1.0 if winner == original_player else 1.0
        
        # 最大手数に達した場合は引き分けとして扱う
        return 0
    
    def get_valid_moves(self, env):
        """有効な手を取得"""
        valid_moves = []
        board = env.board.GetBoardInt()
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board[y][x] == 0:
                    valid_moves.append((x, y))
        return valid_moves

class RandomPlayer:
    """ランダムに手を選ぶプレイヤー"""
    def __init__(self, board_size):
        self.board_size = board_size
    
    def get_action(self, env):
        """有効な手からランダムに選択"""
        valid_moves = []
        board = env.board.GetBoardInt()
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board[y][x] == 0:  # 空きマス
                    valid_moves.append((x, y))
        
        if not valid_moves:
            return None  # 有効な手がない
        
        return random.choice(valid_moves)

class MCTSVsRandomGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Pure MCTS vs Random Player")
        self.master.geometry("1000x700")
        
        # ゲーム設定
        self.board_size = 8  # ボードサイズ
        self.cell_size = 40  # セルサイズ
        self.margin = 50     # マージン
        
        # ゲーム状態
        self.env = None
        self.mcts_player = None
        self.random_player = None
        self.game_over = False
        self.game_running = False
        
        # 統計
        self.mcts_wins = 0
        self.random_wins = 0
        self.draws = 0
        self.total_games = 0
        self.current_first_player = "mcts"  # 現在のゲームの先手プレイヤー
        
        # 詳細統計（先手後手別）
        self.mcts_wins_as_first = 0
        self.mcts_wins_as_second = 0
        self.random_wins_as_first = 0
        self.random_wins_as_second = 0
        
        # GUI要素の初期化
        self.setup_gui()
        
        # プレイヤーの初期化
        self.init_players()
        
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
            text="Pure MCTS vs Random", 
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # ゲーム情報
        self.info_frame = tk.Frame(self.right_frame)
        self.info_frame.pack(fill='x', pady=(0, 20))
        
        self.turn_label = tk.Label(
            self.info_frame, 
            text="準備完了", 
            font=('Arial', 12)
        )
        self.turn_label.pack()
        
        self.status_label = tk.Label(
            self.info_frame, 
            text="ゲーム開始可能", 
            font=('Arial', 10)
        )
        self.status_label.pack()
        
        # Pure MCTS設定
        mcts_frame = tk.Frame(self.right_frame)
        mcts_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(mcts_frame, text="Pure MCTS設定:", font=('Arial', 12, 'bold')).pack()
        
        self.mcts_playout_var = tk.IntVar(value=1000)
        mcts_scale = tk.Scale(
            mcts_frame, 
            from_=50, 
            to=10000, 
            resolution=50,
            orient='horizontal',
            variable=self.mcts_playout_var,
            label="MCTSシミュレーション回数"
        )
        mcts_scale.pack(fill='x')
        
        # Pure MCTS直接入力
        mcts_entry_frame = tk.Frame(mcts_frame)
        mcts_entry_frame.pack(fill='x', pady=(5, 0))
        
        tk.Label(mcts_entry_frame, text="直接入力:", font=('Arial', 10)).pack(side='left')
        
        self.mcts_entry = tk.Entry(mcts_entry_frame, width=10, font=('Arial', 10))
        self.mcts_entry.pack(side='left', padx=(5, 5))
        self.mcts_entry.insert(0, "200")
        
        def update_mcts_from_entry():
            try:
                value = int(self.mcts_entry.get())
                if value > 0:
                    self.mcts_playout_var.set(value)
                    print(f"Pure MCTSプレイアウト回数を更新: {value}")
                    if value > 10000:
                        mcts_scale.config(to=value + 1000)
                else:
                    print("プレイアウト回数は正の値である必要があります")
            except ValueError:
                print("無効な値が入力されました")
        
        def update_mcts_entry_from_scale(*args):
            new_value = self.mcts_playout_var.get()
            self.mcts_entry.delete(0, tk.END)
            self.mcts_entry.insert(0, str(new_value))
        
        tk.Button(mcts_entry_frame, text="適用", command=update_mcts_from_entry, font=('Arial', 9)).pack(side='left')
        self.mcts_playout_var.trace('w', update_mcts_entry_from_scale)
        
        # プレイヤー設定
        player_frame = tk.Frame(self.right_frame)
        player_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(player_frame, text="先手プレイヤー:", font=('Arial', 12, 'bold')).pack()
        
        self.first_player_var = tk.StringVar(value="alternating")
        tk.Radiobutton(
            player_frame, 
            text="Pure MCTS固定", 
            variable=self.first_player_var, 
            value="mcts"
        ).pack()
        tk.Radiobutton(
            player_frame, 
            text="Random固定", 
            variable=self.first_player_var, 
            value="random"
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
        
        self.mcts_wins_label = tk.Label(stats_frame, text="Pure MCTS勝利: 0 (0.0%)", font=('Arial', 10))
        self.mcts_wins_label.pack()
        
        self.random_wins_label = tk.Label(stats_frame, text="Random勝利: 0 (0.0%)", font=('Arial', 10))
        self.random_wins_label.pack()
        
        self.draws_label = tk.Label(stats_frame, text="引き分け: 0 (0.0%)", font=('Arial', 10))
        self.draws_label.pack()
        
        # 詳細統計（先手後手別）
        detail_frame = tk.Frame(self.right_frame)
        detail_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(detail_frame, text="詳細統計:", font=('Arial', 10, 'bold')).pack()
        
        self.mcts_first_label = tk.Label(detail_frame, text="Pure MCTS先手: 0勝", font=('Arial', 9))
        self.mcts_first_label.pack()
        
        self.mcts_second_label = tk.Label(detail_frame, text="Pure MCTS後手: 0勝", font=('Arial', 9))
        self.mcts_second_label.pack()
        
        self.random_first_label = tk.Label(detail_frame, text="Random先手: 0勝", font=('Arial', 9))
        self.random_first_label.pack()
        
        self.random_second_label = tk.Label(detail_frame, text="Random後手: 0勝", font=('Arial', 9))
        self.random_second_label.pack()
    
    def init_players(self):
        """プレイヤーを初期化する"""
        try:
            print("プレイヤー初期化開始...")
            
            # Pure MCTSプレイヤーの初期化
            initial_mcts_playout = self.mcts_playout_var.get()
            self.mcts_player = PureMCTSPlayer(
                self.board_size,
                c_puct=5,
                n_playout=initial_mcts_playout
            )
            
            print(f"Pure MCTSプレイヤー初期化完了: プレイアウト回数 = {initial_mcts_playout}")
            
            # ランダムプレイヤーの初期化
            self.random_player = RandomPlayer(self.board_size)
            
            print("プレイヤー初期化完了")
            self.status_label.config(text="ゲーム開始可能")
            
        except Exception as e:
            error_msg = f"プレイヤーの初期化に失敗しました: {str(e)}"
            print(f"エラー: {error_msg}")
            print("詳細なエラー情報:")
            traceback.print_exc()
            self.status_label.config(text="初期化失敗")
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
        if not self.mcts_player or not self.random_player:
            messagebox.showerror("エラー", "プレイヤーが初期化されていません")
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
                return "mcts"
            else:
                return "random"
        else:
            # 固定設定
            return setting
    
    def reset_stats(self):
        """統計をリセットする"""
        self.mcts_wins = 0
        self.random_wins = 0
        self.draws = 0
        self.total_games = 0
        self.current_first_player = "mcts"
        
        # 詳細統計もリセット
        self.mcts_wins_as_first = 0
        self.mcts_wins_as_second = 0
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
                        time.sleep(0.5)  # ゲーム間の短い間隔
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
        
        # プレイヤーの決定
        first_player_type = self.determine_first_player()
        mcts_is_first = (first_player_type == "mcts")
        
        # 現在の先手プレイヤーを記録
        self.current_first_player = first_player_type
        
        self.master.after(0, lambda: self.turn_label.config(
            text=f"ゲーム{self.total_games + 1} (先手: {'Pure MCTS' if mcts_is_first else 'Random'})"
        ))
        self.master.after(0, self.draw_board)
        
        game_over = False
        move_count = 0
        max_moves = self.board_size * self.board_size
        
        while not game_over and self.game_running and move_count < max_moves:
            current_player_is_mcts = (mcts_is_first and self.env.current_player == 1) or \
                                   (not mcts_is_first and self.env.current_player == 2)
            
            if current_player_is_mcts:
                # Pure MCTSの手番
                self.master.after(0, lambda: self.turn_label.config(text="Pure MCTSの思考中..."))
                
                # Pure MCTS思考時間を動的に更新
                playout_count = self.mcts_playout_var.get()
                if hasattr(self, 'mcts_player'):
                    self.mcts_player.n_playout = playout_count
                    print(f"Pure MCTS思考中: MCTSプレイアウト回数 = {playout_count}")
                    self.master.after(0, lambda: self.status_label.config(
                        text=f"Pure MCTS思考中 (プレイアウト: {playout_count})"
                    ))
                
                action = self.mcts_player.get_action(self.env)
                player_name = "Pure MCTS"
            else:
                # ランダムプレイヤーの手番
                self.master.after(0, lambda: self.turn_label.config(text="Randomの手番"))
                self.master.after(0, lambda: self.status_label.config(text="Random思考中"))
                
                action = self.random_player.get_action(self.env)
                player_name = "Random"
            
            if action is None:
                # 有効な手がない場合は引き分け
                game_over = True
                self.draws += 1
                result = "引き分け"
                break
            
            # 手を実行
            _, reward, done, info = self.env.step(action)
            move_count += 1
            
            # 盤面更新
            self.master.after(0, self.draw_board)
            
            if done:
                game_over = True
                # 勝者の判定
                winner = info.get('win_player', 0)
                if winner == 0:
                    self.draws += 1
                    result = "引き分け"
                elif (mcts_is_first and winner == 1) or (not mcts_is_first and winner == 2):
                    self.mcts_wins += 1
                    if mcts_is_first:
                        self.mcts_wins_as_first += 1
                    else:
                        self.mcts_wins_as_second += 1
                    result = "Pure MCTS勝利"
                else:
                    self.random_wins += 1
                    if not mcts_is_first:
                        self.random_wins_as_first += 1
                    else:
                        self.random_wins_as_second += 1
                    result = "Random勝利"
        
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
            mcts_percentage = (self.mcts_wins / self.total_games) * 100
            random_percentage = (self.random_wins / self.total_games) * 100
            draw_percentage = (self.draws / self.total_games) * 100
            
            self.mcts_wins_label.config(text=f"Pure MCTS勝利: {self.mcts_wins} ({mcts_percentage:.1f}%)")
            self.random_wins_label.config(text=f"Random勝利: {self.random_wins} ({random_percentage:.1f}%)")
            self.draws_label.config(text=f"引き分け: {self.draws} ({draw_percentage:.1f}%)")
            
            # 詳細統計も更新
            self.mcts_first_label.config(text=f"Pure MCTS先手: {self.mcts_wins_as_first}勝")
            self.mcts_second_label.config(text=f"Pure MCTS後手: {self.mcts_wins_as_second}勝")
            self.random_first_label.config(text=f"Random先手: {self.random_wins_as_first}勝")
            self.random_second_label.config(text=f"Random後手: {self.random_wins_as_second}勝")
        else:
            self.mcts_wins_label.config(text="Pure MCTS勝利: 0 (0.0%)")
            self.random_wins_label.config(text="Random勝利: 0 (0.0%)")
            self.draws_label.config(text="引き分け: 0 (0.0%)")
            
            # 詳細統計も初期化
            self.mcts_first_label.config(text="Pure MCTS先手: 0勝")
            self.mcts_second_label.config(text="Pure MCTS後手: 0勝")
            self.random_first_label.config(text="Random先手: 0勝")
            self.random_second_label.config(text="Random後手: 0勝")

def main():
    root = tk.Tk()
    app = MCTSVsRandomGUI(root)
    
    def on_closing():
        app.game_running = False
        if hasattr(app, 'game_thread') and app.game_thread.is_alive():
            app.game_thread.join(timeout=1.0)
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
