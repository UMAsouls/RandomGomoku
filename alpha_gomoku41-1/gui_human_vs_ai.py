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

class HumanPlayer:
    """人間プレイヤー（GUI操作）"""
    def __init__(self, board_size):
        self.board_size = board_size
        self.selected_move = None
        self.move_ready = False
    
    def get_action(self, board):
        """人間の手を待機（GUI操作で設定される）"""
        # この関数は実際には使用されない
        # GUIのクリックイベントでselected_moveが設定される
        return self.selected_move
    
    def reset_move(self):
        """手の選択をリセット"""
        self.selected_move = None
        self.move_ready = False
    
    def set_move(self, x, y):
        """手を設定"""
        self.selected_move = (x, y)
        self.move_ready = True

class HumanVsAIGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("人間プレイヤー vs AlphaGomoku AI")
        self.master.geometry("1000x700")
        
        # ゲーム設定
        self.board_size = 6  # ボードサイズ
        self.cell_size = 40  # セルサイズ
        self.margin = 50     # マージン
        
        # ゲーム状態
        self.env = None
        self.ai_player = None
        self.human_player = None
        self.game_over = False
        self.game_running = False
        self.waiting_for_human = False
        self.human_is_first = True  # 人間が先手かどうか
        
        # 統計
        self.ai_wins = 0
        self.human_wins = 0
        self.draws = 0
        self.total_games = 0
        
        # 詳細統計（先手後手別）
        self.ai_wins_as_first = 0
        self.ai_wins_as_second = 0
        self.human_wins_as_first = 0
        self.human_wins_as_second = 0
        
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
        
        # マウスクリックイベントのバインド
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # コントロールパネル
        self.setup_controls()
        
        # ボードの描画
        self.draw_board()
    
    def setup_controls(self):
        """コントロールパネルをセットアップする"""
        # タイトル
        title_label = tk.Label(
            self.right_frame, 
            text="人間 vs AI", 
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
        
        # AI思考時間設定
        ai_frame = tk.Frame(self.right_frame)
        ai_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(ai_frame, text="AI思考時間:", font=('Arial', 12, 'bold')).pack()
        
        self.ai_playout_var = tk.IntVar(value=1000)
        ai_scale = tk.Scale(
            ai_frame, 
            from_=100, 
            to=10000, 
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
                    if value > 10000:
                        ai_scale.config(to=value + 5000)
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
        
        self.first_player_var = tk.StringVar(value="human")
        tk.Radiobutton(
            player_frame, 
            text="人間プレイヤー", 
            variable=self.first_player_var, 
            value="human"
        ).pack()
        tk.Radiobutton(
            player_frame, 
            text="AIプレイヤー", 
            variable=self.first_player_var, 
            value="ai"
        ).pack()
        
        # コントロールボタン
        button_frame = tk.Frame(self.right_frame)
        button_frame.pack(fill='x', pady=(0, 20))
        
        self.start_button = tk.Button(
            button_frame, 
            text="新しいゲーム", 
            command=self.start_game,
            font=('Arial', 12),
            bg='lightgreen'
        )
        self.start_button.pack(fill='x', pady=2)
        
        self.reset_button = tk.Button(
            button_frame, 
            text="ゲームリセット", 
            command=self.reset_game,
            font=('Arial', 12),
            bg='lightblue',
            state='disabled'
        )
        self.reset_button.pack(fill='x', pady=2)
        
        self.reset_stats_button = tk.Button(
            button_frame, 
            text="統計リセット", 
            command=self.reset_stats,
            font=('Arial', 12)
        )
        self.reset_stats_button.pack(fill='x', pady=2)
        
        # 操作説明
        help_frame = tk.Frame(self.right_frame)
        help_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(help_frame, text="操作方法:", font=('Arial', 10, 'bold')).pack()
        tk.Label(help_frame, text="・盤面をクリックして石を置く", font=('Arial', 9)).pack()
        tk.Label(help_frame, text="・黒石: 先手プレイヤー", font=('Arial', 9)).pack()
        tk.Label(help_frame, text="・白石: 後手プレイヤー", font=('Arial', 9)).pack()
        tk.Label(help_frame, text="・5つ並べると勝利", font=('Arial', 9)).pack()
        
        # 統計表示
        stats_frame = tk.Frame(self.right_frame)
        stats_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(stats_frame, text="統計:", font=('Arial', 12, 'bold')).pack()
        
        self.total_games_label = tk.Label(stats_frame, text="総ゲーム数: 0", font=('Arial', 10))
        self.total_games_label.pack()
        
        self.human_wins_label = tk.Label(stats_frame, text="人間勝利: 0 (0.0%)", font=('Arial', 10))
        self.human_wins_label.pack()
        
        self.ai_wins_label = tk.Label(stats_frame, text="AI勝利: 0 (0.0%)", font=('Arial', 10))
        self.ai_wins_label.pack()
        
        self.draws_label = tk.Label(stats_frame, text="引き分け: 0 (0.0%)", font=('Arial', 10))
        self.draws_label.pack()
        
        # 詳細統計（先手後手別）
        detail_frame = tk.Frame(self.right_frame)
        detail_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(detail_frame, text="詳細統計:", font=('Arial', 10, 'bold')).pack()
        
        self.human_first_label = tk.Label(detail_frame, text="人間先手: 0勝", font=('Arial', 9))
        self.human_first_label.pack()
        
        self.human_second_label = tk.Label(detail_frame, text="人間後手: 0勝", font=('Arial', 9))
        self.human_second_label.pack()
        
        self.ai_first_label = tk.Label(detail_frame, text="AI先手: 0勝", font=('Arial', 9))
        self.ai_first_label.pack()
        
        self.ai_second_label = tk.Label(detail_frame, text="AI後手: 0勝", font=('Arial', 9))
        self.ai_second_label.pack()
    
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
            
            # MCTSプレイヤーの初期化
            initial_playout = self.ai_playout_var.get()
            self.ai_player = MCTSPlayer(
                self.policy_value_net.policy_value_fn,
                c_puct=5,
                n_playout=initial_playout,
                is_selfplay=0
            )
            
            print(f"MCTSプレイヤー初期化完了: 初期プレイアウト回数 = {initial_playout}")
            
            # 人間プレイヤーの初期化
            self.human_player = HumanPlayer(self.board_size)
            
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
    
    def on_canvas_click(self, event):
        """キャンバスクリックイベントハンドラー"""
        if not self.game_running or not self.waiting_for_human or self.game_over:
            return
        
        # クリック位置を盤面座標に変換
        x = round((event.x - self.margin) / self.cell_size)
        y = round((event.y - self.margin) / self.cell_size)
        
        # 盤面の範囲内かチェック
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            # そのマスが空いているかチェック
            board = self.env.board.GetBoardInt()
            if board[y][x] == 0:
                # 人間の手を設定
                self.human_player.set_move(x, y)
                print(f"人間の手: ({x}, {y})")
            else:
                print(f"({x}, {y})は既に石が置かれています")
        else:
            print(f"クリック位置が盤面外です: ({x}, {y})")
    
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
    
    def start_game(self):
        """ゲームを開始する"""
        if not self.ai_player or not self.human_player:
            messagebox.showerror("エラー", "AIモデルが読み込まれていません")
            return
        
        self.game_running = True
        self.game_over = False
        self.waiting_for_human = False
        self.human_player.reset_move()
        self.last_move = None
        
        # プレイヤーの決定
        self.human_is_first = (self.first_player_var.get() == "human")
        
        # 環境の初期化
        self.env = GomokuEnv(board_size=self.board_size)
        
        # ボタンの状態更新
        self.start_button.config(state='disabled')
        self.reset_button.config(state='normal')
        
        # 盤面描画
        self.draw_board()
        
        # ゲームスレッドを開始
        self.game_thread = threading.Thread(target=self.run_game, daemon=True)
        self.game_thread.start()
    
    def reset_game(self):
        """ゲームをリセットする"""
        self.game_running = False
        self.game_over = False
        self.waiting_for_human = False
        self.human_player.reset_move()
        self.last_move = None
        
        # ボタンの状態更新
        self.start_button.config(state='normal')
        self.reset_button.config(state='disabled')
        
        # 表示リセット
        self.turn_label.config(text="リセット完了")
        self.status_label.config(text="新しいゲームを開始してください")
        
        # 盤面クリア
        if hasattr(self, 'env'):
            self.env = None
        self.draw_board()
    
    def reset_stats(self):
        """統計をリセットする"""
        self.ai_wins = 0
        self.human_wins = 0
        self.draws = 0
        self.total_games = 0
        
        # 詳細統計もリセット
        self.ai_wins_as_first = 0
        self.ai_wins_as_second = 0
        self.human_wins_as_first = 0
        self.human_wins_as_second = 0
        
        self.update_stats()
    
    def run_game(self):
        """ゲームを実行する（別スレッド）"""
        try:
            self.play_single_game()
        except Exception as e:
            print(f"ゲーム実行エラー: {str(e)}")
            traceback.print_exc()
            self.master.after(0, lambda: self.status_label.config(text="ゲームエラー"))
        finally:
            self.master.after(0, self.end_game)
    
    def play_single_game(self):
        """一回のゲームを実行する"""
        if not self.game_running:
            return

        # ゲーム開始メッセージ
        self.master.after(0, lambda: self.turn_label.config(
            text=f"ゲーム開始 (先手: {'人間' if self.human_is_first else 'AI'})"
        ))

        move_count = 0
        max_moves = self.board_size * self.board_size

        while not self.game_over and self.game_running and move_count < max_moves:
            current_player_is_human = (self.human_is_first and self.env.current_player == 1) or \
                                    (not self.human_is_first and self.env.current_player == 2)

            if current_player_is_human:
                # 人間の手番
                self.waiting_for_human = True
                self.human_player.reset_move()
                self.master.after(0, lambda: self.turn_label.config(text="あなたの手番"))
                self.master.after(0, lambda: self.status_label.config(text="盤面をクリックして石を置いてください"))

                # 人間の手を待機
                while not self.human_player.move_ready and self.game_running:
                    time.sleep(0.1)

                if not self.game_running:
                    break

                action = self.human_player.get_action(None)
                self.waiting_for_human = False

                if action is None:
                    break

                player_name = "人間"

            else:
                # AIの手番
                self.master.after(0, lambda: self.turn_label.config(text="AIの手番"))
                self.master.after(0, lambda: self.status_label.config(text="AI思考中..."))

                # AI思考時間を動的に更新
                playout_count = self.ai_playout_var.get()
                self.ai_player.n_playout = playout_count
                print(f"AI思考中: MCTSプレイアウト回数 = {playout_count}")

                board = self.env.board.GetBoardInt()
                action = self.find_winning_move(board, self.env.current_player)
                if action is None:
                    action = self.find_double_threat_move(board, self.env.current_player)
                if action is None:
                    action = self.ai_player.get_action(self.env)

                if action is None:
                    break

                player_name = "AI"

            # 手を実行
            _, reward, done, _ = self.env.step(action)
            move_count += 1

            # 最後の手を記録
            self.last_move = action

            # 盤面更新
            self.master.after(0, self.draw_board)

            print(f"{player_name}の手: {action}")

            if done:
                # 勝者の判定
                winner = 3 - self.env.current_player  # 前のプレイヤーが勝者
                if (self.human_is_first and winner == 1) or (not self.human_is_first and winner == 2):
                    self.human_wins += 1
                    if self.human_is_first:
                        self.human_wins_as_first += 1
                    else:
                        self.human_wins_as_second += 1
                    result = "人間の勝利！"
                else:
                    self.ai_wins += 1
                    if not self.human_is_first:
                        self.ai_wins_as_first += 1
                    else:
                        self.ai_wins_as_second += 1
                    result = "AIの勝利！"

                self.game_over = True
                self.total_games += 1

                self.master.after(0, lambda: self.turn_label.config(text=f"ゲーム終了"))
                self.master.after(0, lambda: self.status_label.config(text=result))
                self.master.after(0, self.update_stats)

                # 結果をメッセージボックスで表示
                self.master.after(0, lambda: messagebox.showinfo("ゲーム終了", result))

                break

        # 最大手数に達した場合
        if move_count >= max_moves and not self.game_over:
            self.draws += 1
            self.total_games += 1
            result = "引き分け"

            self.master.after(0, lambda: self.turn_label.config(text="ゲーム終了"))
            self.master.after(0, lambda: self.status_label.config(text=result))
            self.master.after(0, self.update_stats)
            self.master.after(0, lambda: messagebox.showinfo("ゲーム終了", result))

    def find_winning_move(self, board, player):
        """あと一手で勝てる手を返す。なければNone"""
        n_in_row = 4  # 五目並べなら5
        board_size = self.board_size
        for y in range(board_size):
            for x in range(board_size):
                if board[y][x] == 0:
                    board[y][x] = player
                    if self.check_win(x, y, player, n_in_row, board):
                        board[y][x] = 0
                        return (x, y)
                    board[y][x] = 0
        return None

    def find_double_threat_move(self, board, player):
        """あと二手で確実に勝てるダブルリーチの手を返す。なければNone"""
        n_in_row = 4
        board_size = self.board_size
        opponent = 2 if player == 1 else 1
        empty_cells = [(y, x) for y in range(board_size) for x in range(board_size) if board[y][x] == 0]

        # まず、相手が次の一手で勝てるかどうかをチェック
        for oy, ox in empty_cells:
            board[oy][ox] = opponent
            if self.check_win(ox, oy, opponent, n_in_row, board):
                board[oy][ox] = 0
                # 相手が次で勝てるなら自分の2手勝ちは成立しない
                return None
            board[oy][ox] = 0

        for idx1 in range(len(empty_cells)):
            y1, x1 = empty_cells[idx1]
            board[y1][x1] = player
            next_empty = [(y, x) for (y, x) in empty_cells if (y, x) != (y1, x1)]
            win_next = []
            for y2, x2 in next_empty:
                board[y2][x2] = player
                if self.check_win(x2, y2, player, n_in_row, board):
                    win_next.append((y2, x2))
                board[y2][x2] = 0
            if len(win_next) >= 2:
                guaranteed = True
                for block_y, block_x in win_next:
                    board[block_y][block_x] = opponent
                    found = False
                    for y2, x2 in win_next:
                        if (y2, x2) == (block_y, block_x):
                            continue
                        if board[y2][x2] == 0:
                            board[y2][x2] = player
                            if self.check_win(x2, y2, player, n_in_row, board):
                                found = True
                            board[y2][x2] = 0
                    board[block_y][block_x] = 0
                    if not found:
                        guaranteed = False
                        break
                if guaranteed:
                    board[y1][x1] = 0
                    return (x1, y1)
            board[y1][x1] = 0
        return None

    def check_win(self, x, y, player, n_in_row, board):
        """指定位置からn_in_row個並んでいるか判定"""
        board_size = self.board_size
        if x < 0 or y < 0 or x >= board_size or y >= board_size:
            return False
        stone_type = board[y][x]
        if stone_type != player:
            return False
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 右下対角線
            (1, -1)   # 右上対角線
        ]
        for dx, dy in directions:
            count = 1
            nx, ny = x + dx, y + dy
            while 0 <= nx < board_size and 0 <= ny < board_size and board[ny][nx] == player:
                count += 1
                if count >= n_in_row:
                    return True
                nx += dx
                ny += dy
            nx, ny = x - dx, y - dy
            while 0 <= nx < board_size and 0 <= ny < board_size and board[ny][nx] == player:
                count += 1
                if count >= n_in_row:
                    return True
                nx -= dx
                ny -= dy
        return False
    
    def end_game(self):
        """ゲーム終了処理"""
        self.game_running = False
        self.waiting_for_human = False
        self.start_button.config(state='normal')
        self.reset_button.config(state='disabled')
    
    def update_stats(self):
        """統計表示を更新する"""
        self.total_games_label.config(text=f"総ゲーム数: {self.total_games}")
        
        if self.total_games > 0:
            human_percentage = (self.human_wins / self.total_games) * 100
            ai_percentage = (self.ai_wins / self.total_games) * 100
            draw_percentage = (self.draws / self.total_games) * 100
            
            self.human_wins_label.config(text=f"人間勝利: {self.human_wins} ({human_percentage:.1f}%)")
            self.ai_wins_label.config(text=f"AI勝利: {self.ai_wins} ({ai_percentage:.1f}%)")
            self.draws_label.config(text=f"引き分け: {self.draws} ({draw_percentage:.1f}%)")
            
            # 詳細統計も更新
            self.human_first_label.config(text=f"人間先手: {self.human_wins_as_first}勝")
            self.human_second_label.config(text=f"人間後手: {self.human_wins_as_second}勝")
            self.ai_first_label.config(text=f"AI先手: {self.ai_wins_as_first}勝")
            self.ai_second_label.config(text=f"AI後手: {self.ai_wins_as_second}勝")
        else:
            self.human_wins_label.config(text="人間勝利: 0 (0.0%)")
            self.ai_wins_label.config(text="AI勝利: 0 (0.0%)")
            self.draws_label.config(text="引き分け: 0 (0.0%)")
            
            # 詳細統計も初期化
            self.human_first_label.config(text="人間先手: 0勝")
            self.human_second_label.config(text="人間後手: 0勝")
            self.ai_first_label.config(text="AI先手: 0勝")
            self.ai_second_label.config(text="AI後手: 0勝")

def main():
    root = tk.Tk()
    app = HumanVsAIGUI(root)
    
    def on_closing():
        app.game_running = False
        if hasattr(app, 'game_thread') and app.game_thread.is_alive():
            app.game_thread.join(timeout=1.0)
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
