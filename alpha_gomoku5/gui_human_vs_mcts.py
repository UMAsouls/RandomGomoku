import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import threading
import time
import numpy as np
from GomokuEnv import GomokuEnv
from mcts import MCTSPlayer

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

class GomokuGUI:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.cell_size = 30
        self.margin = 30
        
        # ゲーム環境の初期化
        self.env = GomokuEnv(board_size=board_size)
        
        # MCTSプレイヤーの初期化
        dummy_policy = DummyPolicy(board_size)
        self.mcts_player = MCTSPlayer(dummy_policy, c_puct=1.0, n_playout=1000)
        self.mcts_player.set_player_ind(2)  # AIは白（プレイヤー2）
        
        # ゲーム状態
        self.game_over = False
        self.human_turn = True  # 人間が黒（プレイヤー1）で先手
        self.ai_thinking = False
        
        # GUI初期化
        self.root = tk.Tk()
        self.root.title("五目並べ - 人間 vs MCTS")
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
        
        # 設定パネル
        settings_frame = tk.Frame(info_frame)
        settings_frame.pack(side=tk.RIGHT)
        
        tk.Label(settings_frame, text="MCTS思考時間:").pack(side=tk.LEFT)
        self.playout_var = tk.StringVar(value="1000")
        playout_entry = tk.Entry(settings_frame, textvariable=self.playout_var, width=8)
        playout_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        reset_button = tk.Button(settings_frame, text="リセット", command=self.reset_game)
        reset_button.pack(side=tk.LEFT)
        
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
        
        # 星を描画（中央と四隅）
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
                self.status_label.config(text="あなたの勝利です！", fg="blue")
                messagebox.showinfo("ゲーム終了", "あなたの勝利です！")
            elif winner == 2:
                self.status_label.config(text="AIの勝利です！", fg="red")
                messagebox.showinfo("ゲーム終了", "AIの勝利です！")
            else:
                self.status_label.config(text="引き分けです", fg="gray")
                messagebox.showinfo("ゲーム終了", "引き分けです")
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
        
        try:
            # プレイアウト数を更新
            n_playout = int(self.playout_var.get())
            self.mcts_player.mcts._n_playout = n_playout
        except ValueError:
            pass
        
        # AIの手を取得
        action = self.mcts_player.get_action(self.env, temp=1e-3)
        
        if action is None:
            # AIが手を見つけられない場合（ボードが満杯など）
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
                self.status_label.config(text="あなたの勝利です！", fg="blue")
                messagebox.showinfo("ゲーム終了", "あなたの勝利です！")
            elif winner == 2:
                self.status_label.config(text="AIの勝利です！", fg="red")
                messagebox.showinfo("ゲーム終了", "AIの勝利です！")
            else:
                self.status_label.config(text="引き分けです", fg="gray")
                messagebox.showinfo("ゲーム終了", "引き分けです")
            return
        
        # 人間の番に切り替え
        self.human_turn = True
        self.status_label.config(text="あなたの番です（黒）", fg="black")
    
    def handle_ai_no_move(self):
        """AIが手を見つけられない場合の処理"""
        self.hide_progress()
        self.ai_thinking = False
        self.game_over = True
        self.status_label.config(text="引き分けです", fg="gray")
        messagebox.showinfo("ゲーム終了", "引き分けです")
    
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
        self.draw_board()
    
    def run(self):
        """GUIを実行"""
        self.root.mainloop()

if __name__ == "__main__":
    print("五目並べGUI - 人間 vs MCTS")
    print("黒：人間, 白：MCTS AI")
    print("マウスクリックで石を置いてください")
    
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
    game = GomokuGUI(board_size=board_size)
    game.run()
