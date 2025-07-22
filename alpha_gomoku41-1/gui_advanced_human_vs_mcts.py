import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import threading
import time
import numpy as np
from GomokuEnv import GomokuEnv
from mcts import MCTSPlayer

class ImprovedPolicy:
    """改良されたポリシー関数（簡単な評価関数付き）"""
    def __init__(self, board_size=15):
        self.board_size = board_size
    
    def evaluate_position(self, board_state, x, y, player):
        """位置の評価値を計算"""
        if board_state[y][x] != 0:
            return 0
        
        score = 0
        
        # 中央に近いほど高得点
        center = self.board_size // 2
        distance_to_center = abs(x - center) + abs(y - center)
        score += max(0, 10 - distance_to_center)
        
        # 周囲の石の数をカウント
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 0
            # 一方向
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board_state[ny][nx] == player:
                        count += 1
                    else:
                        break
                else:
                    break
            
            # 反対方向
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board_state[ny][nx] == player:
                        count += 1
                    else:
                        break
                else:
                    break
            
            # 連続する石の数に応じてスコアを加算
            if count >= 4:
                score += 1000  # 勝利
            elif count >= 3:
                score += 100
            elif count >= 2:
                score += 20
            elif count >= 1:
                score += 5
        
        return score
    
    def __call__(self, board):
        """
        改良されたポリシーを使用して行動確率と価値を返す
        """
        # 盤面の状態を取得
        board_state = board.GetBoardInt()
        
        # 現在のプレイヤーを推定（簡単な実装）
        black_count = sum(row.count(1) for row in board_state)
        white_count = sum(row.count(2) for row in board_state)
        current_player = 1 if black_count <= white_count else 2
        
        # 有効な手を取得
        valid_moves = []
        scores = []
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board_state[y][x] == 0:
                    action = x + y * self.board_size
                    score = self.evaluate_position(board_state, x, y, current_player)
                    valid_moves.append(action)
                    scores.append(score)
        
        if not valid_moves:
            return [], 0.0
        
        # スコアを正規化して確率に変換
        scores = np.array(scores)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores = scores + 0.1  # 最小値を保証
        scores = scores / scores.sum()
        
        action_priors = [(action, prob) for action, prob in zip(valid_moves, scores)]
        
        # 価値評価（簡単な実装）
        total_score = sum(self.evaluate_position(board_state, x, y, current_player) 
                         for y in range(self.board_size) for x in range(self.board_size))
        leaf_value = np.tanh(total_score / 100.0)  # -1から1の範囲に正規化
        
        return action_priors, leaf_value

class AdvancedGomokuGUI:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.cell_size = 30
        self.margin = 30
        
        # ゲーム環境の初期化
        self.env = GomokuEnv(board_size=board_size)
        
        # MCTSプレイヤーの初期化
        self.policy_type = "improved"  # "random" or "improved"
        self.update_mcts_player()
        
        # ゲーム状態
        self.game_over = False
        self.human_turn = True  # 人間が黒（プレイヤー1）で先手
        self.ai_thinking = False
        
        # 統計情報
        self.human_wins = 0
        self.ai_wins = 0
        self.draws = 0
        
        # GUI初期化
        self.root = tk.Tk()
        self.root.title("五目並べ - 人間 vs 改良MCTS")
        self.root.resizable(False, False)
        
        # メインフレーム
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # 情報パネル
        info_frame = tk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(info_frame, text="あなたの番です（黒）", 
                                   font=("Arial", 12, "bold"))
        self.status_label.pack(side=tk.LEFT)
        
        # 統計表示
        self.stats_label = tk.Label(info_frame, text="人間: 0勝, AI: 0勝, 引き分け: 0", 
                                  font=("Arial", 10))
        self.stats_label.pack(side=tk.RIGHT)
        
        # 設定パネル
        settings_frame = tk.Frame(main_frame)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 第1行：プレイアウト数とポリシー
        settings_row1 = tk.Frame(settings_frame)
        settings_row1.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(settings_row1, text="プレイアウト数:").pack(side=tk.LEFT)
        self.playout_var = tk.StringVar(value="1000")
        playout_entry = tk.Entry(settings_row1, textvariable=self.playout_var, width=8)
        playout_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        tk.Label(settings_row1, text="ポリシー:").pack(side=tk.LEFT)
        self.policy_var = tk.StringVar(value="improved")
        policy_combo = ttk.Combobox(settings_row1, textvariable=self.policy_var, 
                                   values=["random", "improved"], width=10, state="readonly")
        policy_combo.pack(side=tk.LEFT, padx=(5, 15))
        policy_combo.bind("<<ComboboxSelected>>", self.on_policy_change)
        
        # 第2行：MCTS設定
        settings_row2 = tk.Frame(settings_frame)
        settings_row2.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(settings_row2, text="C_PUCT:").pack(side=tk.LEFT)
        self.c_puct_var = tk.StringVar(value="1.4")
        c_puct_entry = tk.Entry(settings_row2, textvariable=self.c_puct_var, width=8)
        c_puct_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        tk.Label(settings_row2, text="並列スレッド数:").pack(side=tk.LEFT)
        self.threads_var = tk.StringVar(value="4")
        threads_entry = tk.Entry(settings_row2, textvariable=self.threads_var, width=8)
        threads_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        # 第3行：ボタン
        button_frame = tk.Frame(settings_frame)
        button_frame.pack(fill=tk.X)
        
        reset_button = tk.Button(button_frame, text="リセット", command=self.reset_game)
        reset_button.pack(side=tk.LEFT, padx=(0, 5))
        
        apply_button = tk.Button(button_frame, text="設定適用", command=self.apply_settings)
        apply_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.hint_button = tk.Button(button_frame, text="ヒント", command=self.show_hint)
        self.hint_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # キャンバス作成
        canvas_size = self.board_size * self.cell_size + 2 * self.margin
        self.canvas = tk.Canvas(main_frame, width=canvas_size, height=canvas_size, 
                               bg="burlywood", highlightthickness=1, highlightbackground="black")
        self.canvas.pack()
        
        # キャンバスにクリックイベントを追加
        self.canvas.bind("<Button-1>", self.on_click)
        
        # 盤面を描画
        self.draw_board()
        
        # プログレスバー（AI思考中表示用）
        self.progress_frame = tk.Frame(main_frame)
        self.progress_label = tk.Label(self.progress_frame, text="AI思考中...")
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # ヒント表示用
        self.hint_positions = []
    
    def update_mcts_player(self):
        """MCTSプレイヤーを更新"""
        if self.policy_type == "improved":
            policy = ImprovedPolicy(self.board_size)
        else:
            policy = DummyPolicy(self.board_size)
        
        c_puct = float(self.c_puct_var.get()) if hasattr(self, 'c_puct_var') else 1.4
        n_playout = int(self.playout_var.get()) if hasattr(self, 'playout_var') else 1000
        
        self.mcts_player = MCTSPlayer(policy, c_puct=c_puct, n_playout=n_playout)
        self.mcts_player.set_player_ind(2)  # AIは白（プレイヤー2）
        
        # 並列設定
        if hasattr(self, 'threads_var'):
            try:
                num_threads = int(self.threads_var.get())
                self.mcts_player.mcts.num_threads = max(1, min(num_threads, 8))
            except ValueError:
                pass
    
    def on_policy_change(self, event=None):
        """ポリシー変更時の処理"""
        self.policy_type = self.policy_var.get()
        self.update_mcts_player()
    
    def apply_settings(self):
        """設定を適用"""
        try:
            self.update_mcts_player()
            messagebox.showinfo("設定", "設定を適用しました")
        except ValueError as e:
            messagebox.showerror("エラー", f"設定エラー: {str(e)}")
    
    def show_hint(self):
        """ヒントを表示"""
        if self.game_over or not self.human_turn or self.ai_thinking:
            return
        
        # AIの推奨手を取得
        self.hint_button.config(state="disabled")
        threading.Thread(target=self.calculate_hint, daemon=True).start()
    
    def calculate_hint(self):
        """ヒントを計算（別スレッド）"""
        temp_mcts = MCTSPlayer(
            ImprovedPolicy(self.board_size) if self.policy_type == "improved" else DummyPolicy(self.board_size),
            c_puct=float(self.c_puct_var.get()),
            n_playout=min(500, int(self.playout_var.get()))  # ヒント用に軽量化
        )
        temp_mcts.set_player_ind(1)  # 人間と同じプレイヤー
        
        action = temp_mcts.get_action(self.env, temp=1e-3)
        if action:
            self.root.after(100, lambda: self.display_hint(action))
        else:
            self.root.after(100, self.hint_failed)
    
    def display_hint(self, action):
        """ヒントを表示"""
        x, y = action
        self.hint_positions = [(x, y)]
        self.draw_board()
        self.hint_button.config(state="normal")
        
        # 3秒後にヒントを消去
        self.root.after(3000, self.clear_hint)
    
    def clear_hint(self):
        """ヒントを消去"""
        self.hint_positions = []
        self.draw_board()
    
    def hint_failed(self):
        """ヒント計算失敗"""
        self.hint_button.config(state="normal")
        messagebox.showinfo("ヒント", "ヒントを計算できませんでした")
    
    def draw_board(self):
        """盤面を描画"""
        self.canvas.delete("all")
        
        # 格子を描画
        for i in range(self.board_size):
            # 縦線
            x = self.margin + i * self.cell_size
            self.canvas.create_line(x, self.margin, x, 
                                  self.margin + (self.board_size - 1) * self.cell_size,
                                  fill="black", width=1)
            # 横線
            y = self.margin + i * self.cell_size
            self.canvas.create_line(self.margin, y, 
                                  self.margin + (self.board_size - 1) * self.cell_size, y,
                                  fill="black", width=1)
        
        # 星を描画
        if self.board_size >= 13:
            star_positions = [
                (3, 3), (3, self.board_size - 4), 
                (self.board_size - 4, 3), (self.board_size - 4, self.board_size - 4),
                (self.board_size // 2, self.board_size // 2)
            ]
            for star_x, star_y in star_positions:
                x = self.margin + star_x * self.cell_size
                y = self.margin + star_y * self.cell_size
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, 
                                      fill="black", outline="black")
        
        # 石を描画
        board_state = self.env.board.GetBoardInt()
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board_state[y][x] != 0:
                    self.draw_stone(x, y, board_state[y][x])
        
        # 最後の手をハイライト
        if self.env.lastmove is not None:
            x, y = self.env.lastmove
            canvas_x = self.margin + x * self.cell_size
            canvas_y = self.margin + y * self.cell_size
            self.canvas.create_rectangle(canvas_x - 10, canvas_y - 10, 
                                       canvas_x + 10, canvas_y + 10,
                                       outline="red", width=2)
        
        # ヒントを描画
        for hint_x, hint_y in self.hint_positions:
            canvas_x = self.margin + hint_x * self.cell_size
            canvas_y = self.margin + hint_y * self.cell_size
            self.canvas.create_oval(canvas_x - 8, canvas_y - 8, 
                                  canvas_x + 8, canvas_y + 8,
                                  outline="blue", width=3, fill="lightblue")
    
    def draw_stone(self, x, y, player):
        """石を描画"""
        canvas_x = self.margin + x * self.cell_size
        canvas_y = self.margin + y * self.cell_size
        radius = self.cell_size // 2 - 2
        
        if player == 1:  # 黒
            color = "black"
            outline = "black"
        else:  # 白
            color = "white"
            outline = "black"
        
        self.canvas.create_oval(canvas_x - radius, canvas_y - radius,
                               canvas_x + radius, canvas_y + radius,
                               fill=color, outline=outline, width=2)
    
    def on_click(self, event):
        """マウスクリックイベント処理"""
        if self.game_over or not self.human_turn or self.ai_thinking:
            return
        
        # ヒントを消去
        self.clear_hint()
        
        # クリック位置を盤面座標に変換
        x = round((event.x - self.margin) / self.cell_size)
        y = round((event.y - self.margin) / self.cell_size)
        
        # 盤面内かチェック
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return
        
        # 空いているマスかチェック
        if self.env.board.GetBoardInt()[y][x] != 0:
            return
        
        # 人間の手を実行
        self.make_human_move(x, y)
    
    def make_human_move(self, x, y):
        """人間の手を実行"""
        _, reward, done, info = self.env.step((x, y))
        self.draw_board()
        
        if done:
            self.game_over = True
            winner = info.get('win_player', 0)
            if winner == 1:
                self.human_wins += 1
                self.status_label.config(text="あなたの勝利です！", fg="blue")
                messagebox.showinfo("ゲーム終了", "あなたの勝利です！")
            elif winner == 2:
                self.ai_wins += 1
                self.status_label.config(text="AIの勝利です！", fg="red")
                messagebox.showinfo("ゲーム終了", "AIの勝利です！")
            else:
                self.draws += 1
                self.status_label.config(text="引き分けです", fg="gray")
                messagebox.showinfo("ゲーム終了", "引き分けです")
            
            self.update_stats()
            return
        
        # AIの番に切り替え
        self.human_turn = False
        self.status_label.config(text="AI思考中...", fg="orange")
        self.show_progress()
        
        # AIの思考を別スレッドで実行
        threading.Thread(target=self.ai_move, daemon=True).start()
    
    def ai_move(self):
        """AIの手を実行（別スレッド）"""
        self.ai_thinking = True
        
        # 設定を適用
        try:
            self.update_mcts_player()
        except ValueError:
            pass
        
        # AIの手を取得
        action = self.mcts_player.get_action(self.env, temp=1e-3)
        
        if action is None:
            self.root.after(100, self.handle_ai_no_move)
            return
        
        # メインスレッドでAIの手を実行
        self.root.after(100, lambda: self.execute_ai_move(action))
    
    def execute_ai_move(self, action):
        """AIの手を実行（メインスレッド）"""
        x, y = action
        _, reward, done, info = self.env.step((x, y))
        
        self.hide_progress()
        self.draw_board()
        self.ai_thinking = False
        
        if done:
            self.game_over = True
            winner = info.get('win_player', 0)
            if winner == 1:
                self.human_wins += 1
                self.status_label.config(text="あなたの勝利です！", fg="blue")
                messagebox.showinfo("ゲーム終了", "あなたの勝利です！")
            elif winner == 2:
                self.ai_wins += 1
                self.status_label.config(text="AIの勝利です！", fg="red")
                messagebox.showinfo("ゲーム終了", "AIの勝利です！")
            else:
                self.draws += 1
                self.status_label.config(text="引き分けです", fg="gray")
                messagebox.showinfo("ゲーム終了", "引き分けです")
            
            self.update_stats()
            return
        
        # 人間の番に切り替え
        self.human_turn = True
        self.status_label.config(text="あなたの番です（黒）", fg="black")
    
    def handle_ai_no_move(self):
        """AIが手を見つけられない場合の処理"""
        self.hide_progress()
        self.ai_thinking = False
        self.game_over = True
        self.draws += 1
        self.status_label.config(text="引き分けです", fg="gray")
        self.update_stats()
        messagebox.showinfo("ゲーム終了", "引き分けです")
    
    def update_stats(self):
        """統計情報を更新"""
        self.stats_label.config(text=f"人間: {self.human_wins}勝, AI: {self.ai_wins}勝, 引き分け: {self.draws}")
    
    def show_progress(self):
        """プログレスバーを表示"""
        self.progress_frame.pack(pady=10)
        self.progress_bar.start()
    
    def hide_progress(self):
        """プログレスバーを非表示"""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
    
    def reset_game(self):
        """ゲームをリセット"""
        self.env = GomokuEnv(board_size=self.board_size)
        self.mcts_player.reset_player()
        self.game_over = False
        self.human_turn = True
        self.ai_thinking = False
        self.status_label.config(text="あなたの番です（黒）", fg="black")
        self.hide_progress()
        self.clear_hint()
        self.draw_board()
    
    def run(self):
        """GUIを実行"""
        self.root.mainloop()

class DummyPolicy:
    """MCTSプレイヤー用のダミーポリシー関数（ニューラルネットワークなし）"""
    def __init__(self, board_size=15):
        self.board_size = board_size
    
    def __call__(self, board):
        """
        ランダムポリシーを使用して行動確率と価値を返す
        """
        # 盤面の状態を取得
        board_state = board.GetBoardInt()
        
        # 有効な手を取得
        valid_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board_state[y][x] == 0:
                    action = x + y * self.board_size
                    valid_moves.append(action)
        
        if not valid_moves:
            return [], 0.0
        
        # 均等な確率を割り当て
        prob = 1.0 / len(valid_moves)
        action_priors = [(action, prob) for action in valid_moves]
        
        # 価値は0（中立）として返す
        leaf_value = 0.0
        
        return action_priors, leaf_value

if __name__ == "__main__":
    print("改良版五目並べGUI - 人間 vs MCTS")
    print("黒：人間, 白：MCTS AI")
    print("機能：ヒント、設定調整、統計表示")
    
    # 盤面サイズを選択
    root = tk.Tk()
    root.withdraw()  # 一時的に非表示
    
    board_size = 15  # デフォルトサイズ
    try:
        size_input = simpledialog.askstring("盤面サイズ", "盤面サイズを入力してください（推奨: 15）", initialvalue="15")
        if size_input:
            board_size = int(size_input)
            if board_size < 5 or board_size > 25:
                messagebox.showwarning("警告", "盤面サイズは5-25の範囲で入力してください。15に設定します。")
                board_size = 15
    except ValueError:
        messagebox.showwarning("警告", "無効な入力です。15に設定します。")
        board_size = 15
    
    root.destroy()
    
    # ゲームGUIを開始
    game = AdvancedGomokuGUI(board_size=board_size)
    game.run()
