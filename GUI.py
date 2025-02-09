import pygame
import sys
from dqn import DQNAgent
import GomokuEnv
from  agent import RandomAgent
from GomokuEnv import Stone
from agent import RuleBasedAgent
from agent import MinimaxAgent
#humanが先行なら"first"、後攻なら"second"を入れてください
env = GomokuEnv.GomokuEnv(train_target="second")
#任意のエージェントを選択してください
opponent_agent = MinimaxAgent()
# opponent_agent = RuleBasedAgent()



# GomokuGUI に run メソッドを追加
class GomokuGUI:
    def __init__(self, state, env, opponent_agent):
        self.state = state
        self.env = env
        self.opponent_agent = opponent_agent
        self.cell_size = 20
        self.board_size = len(state)
        self.screen_size = self.cell_size * self.board_size + 100
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Gomoku")
    
    def draw_board(self):
        self.screen.fill((200, 200, 200))  # 背景色
        font = pygame.font.SysFont(None, 24)
        title = font.render("Gomoku", True, (0, 0, 0))
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
        done = False
        total_reward = 0
        while not done:
            self.draw_board()
            pygame.display.flip()  # 画面を更新
            
            if self.env.current_player == self.env.train_player:
                action = self.get_human_action()
            else:
                action = self.opponent_agent.get_action(self.state, self.env.current_player)
            next_state, reward, done, _ = self.env.step(action)
            self.env.current_player = 3 - self.env.current_player
            self.state = next_state
            total_reward += reward
            print('Total Reward:', total_reward)
            self.env.render()
            
        font = pygame.font.SysFont(None, 30)
        if(self.env.current_player == 1):
            #画面に勝者を表示
            title = font.render("White won", True, (0, 0, 0))
        else:
            title = font.render("Black won", True, (0, 0, 0))
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, self.screen_size // 2))
        pygame.display.flip()
        
        while True:
            
            self.draw_board()
            self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, self.screen_size // 2))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


# === 実行部分 ===
state = env.board.GetBoardInt()
gui = GomokuGUI(state, env, opponent_agent)
gui.run()

