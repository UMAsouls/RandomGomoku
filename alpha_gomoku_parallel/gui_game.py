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

class GomokuGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("五目並べ vs AI")
        self.master.geometry("900x700")
        
        # ゲーム設定
        self.board_size = 8  # ボードサイズ
        self.cell_size = 40  # セルサイズ
        self.margin = 50     # マージン
        
        # ゲーム状態
        self.env = None
        self.ai_player = None
        self.game_over = False
        self.human_turn = True
        self.ai_thinking = False
        
        # GUI要素の初期化
        self.setup_gui()
        
        # AIモデルの読み込み
        self.load_ai_model()
        
        # 新しいゲームを開始
        self.new_game()
    
    def setup_gui(self):
        """GUI要素をセットアップする"""
        # フレーム作成
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # 左側フレーム（ボード）
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side='left', expand=True, fill='both')
        
        # 右側フレーム（コントロール）
        self.right_frame = tk.Frame(self.main_frame, width=200)
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
        self.canvas.bind("<Button-1>", self.on_click)
        
        # コントロールパネル
        self.setup_controls()
        
        # ボードの描画
        self.draw_board()
    
    def setup_controls(self):
        """コントロールパネルをセットアップする"""
        # タイトル
        title_label = tk.Label(
            self.right_frame, 
            text="五目並べ vs AI", 
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # ゲーム情報
        self.info_frame = tk.Frame(self.right_frame)
        self.info_frame.pack(fill='x', pady=(0, 20))
        
        self.turn_label = tk.Label(
            self.info_frame, 
            text="あなたの手番", 
            font=('Arial', 12)
        )
        self.turn_label.pack()
        
        self.status_label = tk.Label(
            self.info_frame, 
            text="準備中...", 
            font=('Arial', 10)
        )
        self.status_label.pack()
        
        # プログレスバー（AI思考中表示用）
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.info_frame, 
            variable=self.progress_var, 
            mode='indeterminate'
        )
        self.progress_bar.pack(fill='x', pady=(10, 0))
        
        # ボタン
        self.button_frame = tk.Frame(self.right_frame)
        self.button_frame.pack(fill='x', pady=(20, 0))
        
        self.new_game_button = tk.Button(
            self.button_frame, 
            text="新しいゲーム", 
            command=self.new_game,
            font=('Arial', 10)
        )
        self.new_game_button.pack(fill='x', pady=(0, 10))
        
        self.hint_button = tk.Button(
            self.button_frame, 
            text="ヒント", 
            command=self.show_hint,
            font=('Arial', 10)
        )
        self.hint_button.pack(fill='x', pady=(0, 10))
        
        # 難易度設定
        difficulty_frame = tk.Frame(self.right_frame)
        difficulty_frame.pack(fill='x', pady=(20, 0))
        
        tk.Label(difficulty_frame, text="AI思考時間:", font=('Arial', 10)).pack()
        
        self.difficulty_var = tk.StringVar(value="1000")
        difficulty_scale = tk.Scale(
            difficulty_frame, 
            from_=100, 
            to=3000, 
            orient=tk.HORIZONTAL,
            variable=self.difficulty_var,
            resolution=100
        )
        difficulty_scale.pack(fill='x')
        
        # ゲーム統計
        stats_frame = tk.Frame(self.right_frame)
        stats_frame.pack(fill='x', pady=(20, 0))
        
        tk.Label(stats_frame, text="統計:", font=('Arial', 12, 'bold')).pack()
        
        self.wins_label = tk.Label(stats_frame, text="勝利: 0", font=('Arial', 10))
        self.wins_label.pack()
        
        self.losses_label = tk.Label(stats_frame, text="敗北: 0", font=('Arial', 10))
        self.losses_label.pack()
        
        self.draws_label = tk.Label(stats_frame, text="引き分け: 0", font=('Arial', 10))
        self.draws_label.pack()
        
        # 統計の初期化
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
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
                model_file="best_policy.model",
                use_gpu=use_gpu,
                env=temp_env
            )
            
            # MCTSプレイヤーの初期化
            self.ai_player = MCTSPlayer(
                self.policy_value_net.policy_value_fn,
                c_puct=5,
                n_playout=int(self.difficulty_var.get()),
                is_selfplay=0
            )
            
            print("AIモデル読み込み完了")
            self.status_label.config(text="AIモデル読み込み完了")
            
        except Exception as e:
            error_msg = f"AIモデルの読み込みに失敗しました: {str(e)}"
            print(f"エラー: {error_msg}")
            print("詳細なエラー情報:")
            traceback.print_exc()
            messagebox.showerror("エラー", error_msg)
            self.status_label.config(text="AIモデル読み込み失敗")
    
    def new_game(self):
        """新しいゲームを開始する"""
        print("新しいゲームを開始します")
        self.env = GomokuEnv(board_size=self.board_size)
        self.game_over = False
        self.human_turn = True
        self.ai_thinking = False
        
        # PolicyValueNetの環境を更新
        if self.policy_value_net:
            self.policy_value_net.env = self.env
            print("PolicyValueNetの環境を更新しました")
        
        # AIプレイヤーの設定を更新
        if self.ai_player:
            self.ai_player.n_playout = int(self.difficulty_var.get())
            print(f"AI思考回数: {self.ai_player.n_playout}")
        
        # 表示を更新
        self.update_display()
        self.status_label.config(text="ゲーム開始")
        self.turn_label.config(text="あなたの手番")
        self.progress_bar.stop()
        
        # ボードを再描画
        self.draw_board()
        self.draw_stones()
        print("新しいゲームの準備完了")
    
    def draw_board(self):
        """ボードのグリッドを描画する"""
        self.canvas.delete("grid")
        
        for i in range(self.board_size):
            # 縦線
            x = self.margin + i * self.cell_size
            self.canvas.create_line(
                x, self.margin, 
                x, self.margin + (self.board_size - 1) * self.cell_size,
                tags="grid", width=1, fill="black"
            )
            
            # 横線
            y = self.margin + i * self.cell_size
            self.canvas.create_line(
                self.margin, y,
                self.margin + (self.board_size - 1) * self.cell_size, y,
                tags="grid", width=1, fill="black"
            )
        
        # 星（目印）を描画
        star_positions = []
        if self.board_size >= 15:
            # 19路盤の場合
            star_positions = [(3, 3), (3, 15), (15, 3), (15, 15), (9, 9)]
        elif self.board_size >= 9:
            # 13路盤以上の場合
            mid = self.board_size // 2
            star_positions = [(2, 2), (2, self.board_size-3), 
                            (self.board_size-3, 2), (self.board_size-3, self.board_size-3), 
                            (mid, mid)]
        
        for row, col in star_positions:
            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                self.canvas.create_oval(
                    x-3, y-3, x+3, y+3,
                    fill="black", tags="grid"
                )
    
    def draw_stones(self):
        """石を描画する"""
        self.canvas.delete("stone")
        self.canvas.delete("last_move")
        
        if self.env is None:
            return
        
        board = self.env.board.GetBoardInt()
        stone_radius = self.cell_size // 2 - 2
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] != 0:
                    x = self.margin + col * self.cell_size
                    y = self.margin + row * self.cell_size
                    
                    # 石の描画
                    color = "black" if board[row][col] == 1 else "white"
                    outline_color = "gray" if color == "white" else "black"
                    
                    self.canvas.create_oval(
                        x - stone_radius, y - stone_radius,
                        x + stone_radius, y + stone_radius,
                        fill=color, outline=outline_color, width=2,
                        tags="stone"
                    )
        
        # 最後の手を強調表示
        if self.env.lastmove is not None:
            last_col, last_row = self.env.lastmove  # (x, y) = (col, row)の順序
            x = self.margin + last_col * self.cell_size
            y = self.margin + last_row * self.cell_size
            
            self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                fill="red", tags="last_move"
            )
    
    def on_click(self, event):
        """マウスクリック時の処理"""
        if self.game_over or not self.human_turn or self.ai_thinking:
            return
        
        # クリック位置をボード座標に変換
        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)
        
        # ボード範囲内かチェック
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return
        
        # 空いているセルかチェック
        board = self.env.board.GetBoardInt()
        if board[row][col] != 0:
            return
        
        # 人間の手を実行
        self.make_human_move(row, col)
    
    def make_human_move(self, row, col):
        """人間の手を実行する"""
        try:
            print(f"人間の手: ({row}, {col})")
            # 手を実行
            _, reward, done, info = self.env.step((col, row))
            
            # 表示を更新
            self.draw_stones()
            
            # ゲーム終了チェック
            if done:
                print(f"ゲーム終了 - 報酬: {reward}, 情報: {info}")
                self.handle_game_end(reward, info)
                return
            
            # AIの手番に変更
            self.human_turn = False
            self.turn_label.config(text="AIの手番")
            self.progress_bar.start(10)
            self.ai_thinking = True
            
            print("AIの思考開始...")
            # AIの手を別スレッドで実行
            threading.Thread(target=self.make_ai_move, daemon=True).start()
            
        except Exception as e:
            error_msg = f"手の実行中にエラーが発生しました: {str(e)}"
            print(f"エラー: {error_msg}")
            print("詳細なエラー情報:")
            traceback.print_exc()
            messagebox.showerror("エラー", error_msg)
    
    def make_ai_move(self):
        """AIの手を実行する（別スレッドで実行）"""
        try:
            print("AI思考中...")
            # AIの手を取得
            action = self.ai_player.get_action(self.env)
            print(f"AIの選択した手: {action}")
            
            # メインスレッドで手を実行
            self.master.after(0, self.execute_ai_move, action)
            
        except Exception as e:
            error_msg = f"AIの思考中にエラーが発生しました: {str(e)}"
            print(f"エラー: {error_msg}")
            print("詳細なエラー情報:")
            traceback.print_exc()
            self.master.after(0, lambda: messagebox.showerror("エラー", error_msg))
            self.master.after(0, self.reset_turn)
    
    def execute_ai_move(self, action):
        """AIの手を実際に実行する（メインスレッドで実行）"""
        try:
            if action is None:
                print("AIの手がNone - ゲーム終了（引き分け）")
                self.handle_game_end(0, {"draw": True})
                return
            
            row, col = action
            print(f"AIの手を実行: ({row}, {col})")
            
            # 手が有効かチェック
            board = self.env.board.GetBoardInt()
            if board[row][col] != 0:
                print(f"警告: AIが無効な手を選択しました ({row}, {col}) - 既に石があります")
                # 有効な手をランダムに選択
                valid_moves = []
                for r in range(self.board_size):
                    for c in range(self.board_size):
                        if board[r][c] == 0:
                            valid_moves.append((r, c))
                
                if valid_moves:
                    import random
                    row, col = random.choice(valid_moves)
                    print(f"代替手を選択: ({row}, {col})")
                else:
                    print("有効な手がありません - 引き分け")
                    self.handle_game_end(0, {"draw": True})
                    return
            
            _, reward, done, info = self.env.step((col, row))
            
            # 表示を更新
            self.draw_stones()
            self.progress_bar.stop()
            self.ai_thinking = False
            
            # ゲーム終了チェック
            if done:
                print(f"ゲーム終了 - 報酬: {reward}, 情報: {info}")
                if "invalid_action" in info:
                    print("無効な手によるゲーム終了")
                    self.handle_game_end(0, {"draw": True})
                else:
                    self.handle_game_end(-reward, info)  # AIが勝った場合は負の報酬
                return
            
            # 人間の手番に戻す
            self.human_turn = True
            self.turn_label.config(text="あなたの手番")
            print("人間の手番に戻りました")
            
        except Exception as e:
            error_msg = f"AIの手の実行中にエラーが発生しました: {str(e)}"
            print(f"エラー: {error_msg}")
            print("詳細なエラー情報:")
            traceback.print_exc()
            messagebox.showerror("エラー", error_msg)
            self.reset_turn()
    
    def reset_turn(self):
        """手番をリセットする"""
        self.progress_bar.stop()
        self.ai_thinking = False
        self.human_turn = True
        self.turn_label.config(text="あなたの手番")
    
    def handle_game_end(self, reward, info):
        """ゲーム終了時の処理"""
        self.game_over = True
        self.progress_bar.stop()
        self.ai_thinking = False
        
        # 結果の判定
        if "draw" in info or reward == 0:
            result = "引き分け"
            self.draws += 1
            print("ゲーム結果: 引き分け")
        elif reward > 0:
            result = "あなたの勝利！"
            self.wins += 1
            print("ゲーム結果: プレイヤーの勝利")
        else:
            result = "AIの勝利"
            self.losses += 1
            print("ゲーム結果: AIの勝利")
        
        # 統計を更新
        self.update_stats()
        
        # 結果を表示
        self.turn_label.config(text="ゲーム終了")
        self.status_label.config(text=result)
        
        print(f"統計 - 勝利: {self.wins}, 敗北: {self.losses}, 引き分け: {self.draws}")
        messagebox.showinfo("ゲーム終了", result)
    
    def update_stats(self):
        """統計表示を更新する"""
        self.wins_label.config(text=f"勝利: {self.wins}")
        self.losses_label.config(text=f"敗北: {self.losses}")
        self.draws_label.config(text=f"引き分け: {self.draws}")
    
    def update_display(self):
        """表示を更新する"""
        if self.env is None:
            return
        
        self.draw_stones()
    
    def show_hint(self):
        """ヒントを表示する"""
        if self.game_over or not self.human_turn or self.ai_thinking:
            return
        
        try:
            print("ヒント計算開始...")
            self.status_label.config(text="ヒント計算中...")
            
            # 軽量なAIでヒントを計算
            hint_player = MCTSPlayer(
                self.policy_value_net.policy_value_fn,
                c_puct=5,
                n_playout=200,  # 軽量計算
                is_selfplay=0
            )
            
            def calculate_hint():
                try:
                    action = hint_player.get_action(self.env)
                    print(f"ヒント結果: {action}")
                    if action:
                        row, col = action
                        self.master.after(0, lambda: self.show_hint_result(row, col))
                    else:
                        print("ヒントが見つかりませんでした")
                        self.master.after(0, lambda: self.status_label.config(text="ヒントが見つかりません"))
                except Exception as e:
                    print(f"ヒント計算エラー: {str(e)}")
                    print("詳細なエラー情報:")
                    traceback.print_exc()
                    self.master.after(0, lambda: self.status_label.config(text="ヒント計算失敗"))
            
            threading.Thread(target=calculate_hint, daemon=True).start()
            
        except Exception as e:
            print(f"ヒント機能エラー: {str(e)}")
            print("詳細なエラー情報:")
            traceback.print_exc()
            self.status_label.config(text="ヒント機能エラー")
    
    def show_hint_result(self, row, col):
        """ヒント結果を表示する"""
        # ヒント位置をハイライト
        self.canvas.delete("hint")
        
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        
        self.canvas.create_oval(
            x - 15, y - 15, x + 15, y + 15,
            outline="blue", width=3, tags="hint"
        )
        
        self.status_label.config(text=f"推奨手: ({row+1}, {col+1})")
        
        # 3秒後にヒントを消す
        self.master.after(3000, lambda: self.canvas.delete("hint"))

def main():
    print("五目並べGUI アプリケーション開始")
    print("=" * 50)
    root = tk.Tk()
    game = GomokuGUI(root)
    print("GUIの初期化完了")
    print("=" * 50)
    root.mainloop()
    print("アプリケーション終了")

if __name__ == "__main__":
    main()