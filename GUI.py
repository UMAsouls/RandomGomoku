import pygame
import sys
# from dqn import DQNAgent
# from pytorch_dqn import MCTSAgent
import GomokuEnv
from  agent import RandomAgent
from GomokuEnv import Stone
# from agent import RuleBasedAgent
# from agent import MinimaxAgent
from rapfi_agent import RapfiAgent  # 新しいRapfiAgentをインポート
from adp_agent import ADPAgent  # 追加: ADPエージェントをインポート
from alpha_gomoku import DualNetwork  # DualNetworkをインポート
import os
import torch

#humanが先行なら"first"、後攻なら"second"を入れてください
env = GomokuEnv.GomokuEnv(board_size=7, train_target="second")  # ボードサイズを15x15に固定
# モデルファイルのデフォルトパスを設定
DEFAULT_ADP_MODEL_PATH = 'models/alpha_gomoku_7_iter70_after_selfplay_20250610_102724.pth'

# モデルのロード可能状態を確認する変数
adp_model_loadable = True

# モデルファイルのパスを確認
adp_model_path = DEFAULT_ADP_MODEL_PATH
if not os.path.exists(adp_model_path):
    print(f"警告: デフォルトのモデルファイル {adp_model_path} が見つかりません。")
    adp_model_loadable = False

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ADPAgent初期化パラメータを設定
adp_agent_params = {
    "model_path": adp_model_path,
    "use_dual_network": True,  # DualNetworkを使用するフラグを追加
    "board_size": 7           # DualNetworkのボードサイズ
}

# ADPモデルの動作確認
try:
    # DualNetworkモデルをテスト
    test_model = DualNetwork(board_size=7).to(device)
    test_model.load_state_dict(torch.load(adp_model_path, map_location=device))
    
    # テスト用のADPAgentを作成
    test_agent = ADPAgent(**adp_agent_params)
    
    # モデルの読み込み確認
    if hasattr(test_agent, 'model_loaded') and not test_agent.model_loaded:
        adp_model_loadable = False
except Exception as e:
    print(f"ADPモデルのテスト中にエラーが発生しました: {e}")
    adp_model_loadable = False

#任意のエージェントを選択してください
# opponent_agent = MCTSAgent(simulations=5000)  # シミュレーション回数は調整可能
# opponent_agent = RuleBasedAgent()
# opponent_agent = DQNAgent()
# ADPモデルが読み込めない場合はRandomAgentをデフォルトに
opponent_agent = ADPAgent(**adp_agent_params) if adp_model_loadable else RandomAgent()


# GomokuGUI に run メソッドを追加
class GomokuGUI:
    def __init__(self, state, env, opponent_agent):
        self.state = state
        self.env = env
        self.opponent_agent = opponent_agent
        self.cell_size = 30  # セルサイズを大きくして見やすく
        self.board_size = 7  # ボードサイズを15x15に固定
        self.screen_size = self.cell_size * self.board_size + 100
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Gomoku 15x15")
    
    def choose_opponent(self):
        """対戦相手AIを選択するUI"""
        self.screen.fill((220, 179, 92))
        font = pygame.font.SysFont(None, 30)
        title = font.render("対戦相手のAIを選んでください", True, (0, 0, 0))
        random_button = font.render("ランダムAI", True, (0, 0, 0))
        
        # ADPモデルのロード状態に応じてボタンの色を変更
        adp_button_color = (0, 0, 0) if adp_model_loadable else (180, 0, 0)
        adp_button_text = "ADP AI" if adp_model_loadable else "ADP AI (利用不可)"
        adp_button = font.render(adp_button_text, True, adp_button_color)
        
        rapfi_button = font.render("Rapfi AI", True, (0, 0, 0))
        
        # モデルロード失敗時の警告メッセージ
        warning_font = pygame.font.SysFont(None, 24)
        warning_text = warning_font.render("※ADPモデルをロードできませんでした", True, (180, 0, 0))
        
        title_rect = title.get_rect(center=(self.screen_size//2, self.screen_size//2 - 100))
        random_rect = pygame.Rect(self.screen_size//2 - 150, self.screen_size//2, 100, 40)
        adp_rect = pygame.Rect(self.screen_size//2 - 50, self.screen_size//2, 100, 40)
        rapfi_rect = pygame.Rect(self.screen_size//2 + 50, self.screen_size//2, 100, 40)
        
        self.screen.blit(title, title_rect)
        pygame.draw.rect(self.screen, (180, 180, 180), random_rect)
        pygame.draw.rect(self.screen, (180, 180, 180), adp_rect)
        pygame.draw.rect(self.screen, (180, 180, 180), rapfi_rect)
        
        self.screen.blit(random_button, random_button.get_rect(center=random_rect.center))
        self.screen.blit(adp_button, adp_button.get_rect(center=adp_rect.center))
        self.screen.blit(rapfi_button, rapfi_button.get_rect(center=rapfi_rect.center))
        
        # モデルロード失敗時に警告を表示
        if not adp_model_loadable:
            warning_rect = warning_text.get_rect(center=(self.screen_size//2, self.screen_size//2 + 60))
            self.screen.blit(warning_text, warning_rect)
        
        pygame.display.flip()
        
        waiting = True
        selected_agent = None
        
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = event.pos
                    if random_rect.collidepoint(mouse_pos):
                        selected_agent = RandomAgent()
                        waiting = False
                    elif adp_rect.collidepoint(mouse_pos):
                        if adp_model_loadable:
                            selected_agent = ADPAgent(model_path=adp_model_path)
                            waiting = False
                        else:
                            # モデルがロードできない場合は警告を強調
                            warning_text = warning_font.render("※ADPモデルをロードできません。他のAIを選択してください", True, (230, 0, 0))
                            warning_rect = warning_text.get_rect(center=(self.screen_size//2, self.screen_size//2 + 60))
                            self.screen.blit(warning_text, warning_rect)
                            pygame.display.flip()
                    elif rapfi_rect.collidepoint(mouse_pos):
                        selected_agent = RapfiAgent()
                        waiting = False
        
        return selected_agent
    
    def draw_board(self):
        self.screen.fill((220, 179, 92))  # 碁盤の背景色
        font = pygame.font.SysFont(None, 24)
        title = font.render("Gomoku 7*7", True, (0, 0, 0))
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 10))
        
        # 碁盤の線を描画
        for i in range(self.board_size):
            # 縦線
            pygame.draw.line(self.screen, (0, 0, 0), (50 + i * self.cell_size, 50), (50 + i * self.cell_size, 50 + self.cell_size * (self.board_size-1)))
            # 横線
            pygame.draw.line(self.screen, (0, 0, 0), (50, 50 + i * self.cell_size), (50 + self.cell_size * (self.board_size - 1), 50 + i * self.cell_size))
        
        # 星の位置（3-3, 3-11, 7-7, 11-3, 11-11）に点を描画
        star_points = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11), (11, 3), (11, 7), (11, 11)]
        for x, y in star_points:
            pygame.draw.circle(self.screen, (0, 0, 0), (50 + x * self.cell_size, 50 + y * self.cell_size), 4)
        
        # 石を描画
        for i in range(self.board_size):
            for j in range(self.board_size):
                x = 50 + i * self.cell_size
                y = 50 + j * self.cell_size
                if self.state[j][i] == 1:  # 黒石
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), self.cell_size // 2 - 2)
                elif self.state[j][i] == 2:  # 白石
                    pygame.draw.circle(self.screen, (255, 255, 255), (x, y), self.cell_size // 2 - 2)
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), self.cell_size // 2 - 2, 1)

    
    def get_human_action(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 左クリック
                    x, y = event.pos
                    board_x = round((x - 50) / self.cell_size)
                    board_y = round((y - 50) / self.cell_size)
                    if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                        if self.state[board_y][board_x] == 0:
                            return (board_x, board_y)
    
    def run(self):
        # 対戦相手のエージェントを選択するUI
        self.opponent_agent = self.choose_opponent()
        
        # 対戦相手のエージェントをプリント
        if isinstance(self.opponent_agent, RapfiAgent):
            print("Opponent Agent: Rapfi")
        elif isinstance(self.opponent_agent, ADPAgent):
            print("Opponent Agent: ADP")
        else:
            print("Opponent Agent: Random")
            
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

