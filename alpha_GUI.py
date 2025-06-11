import pygame
import sys
import os
import torch
import numpy as np
from alpha_gomoku import DualNetwork, MCTS
import GomokuEnv

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlphaGomokuGUI:
    def __init__(self, board_size=19, model_path='models/alpha_gomoku_19.pth'):
        self.board_size = board_size
        self.model_path = model_path
        self.cell_size = 20
        self.screen_size = self.cell_size * self.board_size + 100
        
        # モデルの初期化とロード
        self.model = DualNetwork(board_size).to(device)
        if os.path.exists(model_path):
            try:
                print(f"モデルをロードしています: {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"モデルを読み込みました: {model_path}")
            except Exception as e:
                print(f"モデルの読み込みに失敗しました: {str(e)}")
                print("新しいモデルを初期化します。")
        else:
            print(f"モデルが見つかりません: {model_path}")
            
        self.model.eval()  # 評価モード
        
        # MCTSの初期化
        self.mcts = MCTS(self.model, num_simulations=800)  # 実戦時はシミュレーション回数を増やす
        
        # 環境の初期化
        self.env = GomokuEnv.GomokuEnv(board_size=board_size)
        self.state = self.env.board.GetBoardInt()
        
        # Pygameの初期化
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("AlphaGomoku")
    
    def draw_board(self):
        self.screen.fill((200, 200, 200))  # 背景色
        font = pygame.font.SysFont(None, 24)
        title = font.render("AlphaGomoku", True, (0, 0, 0))
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 10))
        
        # 碁盤の線を描画
        for i in range(self.board_size):
            # 縦線
            pygame.draw.line(self.screen, (0, 0, 0), (30 + i * self.cell_size, 50), (30 + i * self.cell_size, 40 + self.cell_size * (self.board_size-0.5)))
            # 横線
            pygame.draw.line(self.screen, (0, 0, 0), (30, 50 + i * self.cell_size), (20 + self.cell_size * (self.board_size - 0.5), 50 + i * self.cell_size))
        
        # 石を描画
        for i in range(self.board_size):
            for j in range(self.board_size):
                x = 20 * i + 20
                y = 20 * j + 40
                if self.state[j][i] == 1:  # 黒石
                    pygame.draw.circle(self.screen, (0, 0, 0), (x + 10, y + 10), 8)
                elif self.state[j][i] == 2:  # 白石 
                    pygame.draw.circle(self.screen, (255, 255, 255), (x + 10, y + 10), 8)
                    pygame.draw.circle(self.screen, (0, 0, 0), (x + 10, y + 10), 8, 1)
    
    def get_human_action(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 左クリック
                    x, y = event.pos
                    board_x = (x - 20) // 20
                    board_y = (y - 40) // 20
                    if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                        if self.state[board_y][board_x] == 0:
                            return (board_x, board_y)
    
    def run(self):
        human_first = self.choose_player()
        done = False
        
        # 人間が後手の場合、AIが先に打つ
        if not human_first:
            self.draw_board()
            pygame.display.flip()
            
            # AIの手番
            mcts_policy = self.mcts.search(np.array(self.state))
            action_idx = np.argmax(mcts_policy)
            action = (action_idx % self.board_size, action_idx // self.board_size)
            next_state, reward, done, _ = self.env.step(action)
            self.state = next_state.cpu().numpy()
        
        # メインゲームループ
        while not done:
            self.draw_board()
            pygame.display.flip()
            
            # 人間の手番
            action = self.get_human_action()
            next_state, reward, done, _ = self.env.step(action)
            self.state = next_state.cpu().numpy()
            
            if done:
                self.display_winner("あなたの勝ちです！")
                break
            
            self.draw_board()
            pygame.display.flip()
            
            # AIの手番
            mcts_policy = self.mcts.search(np.array(self.state))
            action_idx = np.argmax(mcts_policy)
            action = (action_idx % self.board_size, action_idx // self.board_size)
            next_state, reward, done, _ = self.env.step(action)
            self.state = next_state.cpu().numpy()
            
            if done:
                self.display_winner("AIの勝ちです!")
    
    def choose_player(self):
        """先手か後手かを選択するUI"""
        self.screen.fill((200, 200, 200))
        font = pygame.font.SysFont(None, 30)
        title = font.render("先手(黒)と後手(白)どちらにしますか？", True, (0, 0, 0))
        first_button = font.render("先手(黒)", True, (0, 0, 0))
        second_button = font.render("後手(白)", True, (0, 0, 0))
        
        title_rect = title.get_rect(center=(self.screen_size//2, self.screen_size//2 - 50))
        first_rect = pygame.Rect(self.screen_size//2 - 100, self.screen_size//2, 80, 40)
        second_rect = pygame.Rect(self.screen_size//2 + 20, self.screen_size//2, 80, 40)
        
        self.screen.blit(title, title_rect)
        pygame.draw.rect(self.screen, (180, 180, 180), first_rect)
        pygame.draw.rect(self.screen, (180, 180, 180), second_rect)
        
        self.screen.blit(first_button, first_button.get_rect(center=first_rect.center))
        self.screen.blit(second_button, second_button.get_rect(center=second_rect.center))
        
        pygame.display.flip()
        
        waiting_for_choice = True
        choice = False  # デフォルト値
        
        while waiting_for_choice:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = event.pos
                    if first_rect.collidepoint(mouse_pos):
                        choice = True  # 先手(黒)
                        waiting_for_choice = False
                    elif second_rect.collidepoint(mouse_pos):
                        choice = False  # 後手(白)
                        waiting_for_choice = False
        
        return choice
    
    def display_winner(self, message):
        """勝者を表示"""
        self.draw_board()
        font = pygame.font.SysFont(None, 30)
        winner_text = font.render(message, True, (255, 0, 0))
        text_rect = winner_text.get_rect(center=(self.screen_size//2, self.screen_size//2))
        self.screen.blit(winner_text, text_rect)
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False

# 実行部分
gui = AlphaGomokuGUI(board_size=7, model_path='models/alpha_gomoku_8_iter138_trained_20250611_133814.pth')
gui.run()