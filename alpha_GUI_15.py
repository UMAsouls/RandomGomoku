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
    def __init__(self, board_size=15, model_path='models/alpha_gomoku_15.pth'):
        self.board_size = board_size
        self.model_path = model_path
        self.cell_size = 40  # セルサイズを40に拡大（元は20）
        self.margin = 120    # マージンも増やして余白を適切に確保
        self.screen_size = self.cell_size * self.board_size + self.margin
        
        # 最後に打った手の位置を追跡（初期値はNone）
        self.last_move = None
        
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
    
    # 日本語対応フォントを取得する関数
    def get_jpn_font(self, size):
        # 日本語対応フォントを優先順位をつけて試す
        font_names = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'MS Mincho', 'Arial Unicode MS']
        for font_name in font_names:
            try:
                return pygame.font.SysFont(font_name, size)
            except:
                continue
        # フォールバック: デフォルトのフォント
        return pygame.font.SysFont(None, size)
    
    def draw_board(self):
        self.screen.fill((200, 200, 200))  # 背景色
        font = self.get_jpn_font(30)  # フォントサイズも大きく
        title = font.render("AlphaGomoku", True, (0, 0, 0))
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 20))
        
        # 現在の手番を表示
        current_player_text = "黒の番（あなた）" if self.env.current_player == 1 else "白の番（AI）"
        if not hasattr(self, 'human_is_black'):
            # 初期化前はデフォルト表示
            current_player_text = "手番情報"
        elif self.human_is_black and self.env.current_player == 2:
            current_player_text = "白の番（AI）"
        elif not self.human_is_black and self.env.current_player == 1:
            current_player_text = "黒の番（AI）"
        elif self.human_is_black and self.env.current_player == 1:
            current_player_text = "黒の番（あなた）"
        else:
            current_player_text = "白の番（あなた）"
            
        player_font = self.get_jpn_font(24)
        player_text = player_font.render(current_player_text, True, (0, 0, 0))
        self.screen.blit(player_text, (20, 20))
        
        # 碁盤の線を描画（調整されたサイズで）
        board_offset_x = self.margin // 2
        board_offset_y = 70  # 上部の余白
        
        for i in range(self.board_size):
            # 縦線
            pygame.draw.line(
                self.screen, (0, 0, 0),
                (board_offset_x + i * self.cell_size, board_offset_y),
                (board_offset_x + i * self.cell_size, board_offset_y + (self.board_size-1) * self.cell_size)
            )
            # 横線
            pygame.draw.line(
                self.screen, (0, 0, 0),
                (board_offset_x, board_offset_y + i * self.cell_size),
                (board_offset_x + (self.board_size-1) * self.cell_size, board_offset_y + i * self.cell_size)
            )
        
        # 石を描画（調整されたサイズで）
        stone_radius = int(self.cell_size * 0.4)  # セルサイズの40%を石の半径に
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                x = board_offset_x + i * self.cell_size
                y = board_offset_y + j * self.cell_size
                
                if self.state[j][i] == 1:  # 黒石
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), stone_radius)
                elif self.state[j][i] == 2:  # 白石 
                    pygame.draw.circle(self.screen, (255, 255, 255), (x, y), stone_radius)
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), stone_radius, 1)
        
        # 最後に打った手がある場合、赤い十字線を描画
        if self.last_move:
            last_x, last_y = self.last_move
            x = board_offset_x + last_x * self.cell_size
            y = board_offset_y + last_y * self.cell_size
            cross_size = stone_radius + 5  # 十字線のサイズ
            
            # 水平線
            pygame.draw.line(self.screen, (255, 0, 0), (x - cross_size, y), (x + cross_size, y), 2)
            # 垂直線
            pygame.draw.line(self.screen, (255, 0, 0), (x, y - cross_size), (x, y + cross_size), 2)
    
    # 空きマスを取得
    def get_valid_moves(self):
        valid_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.state[y][x] == 0:
                    valid_moves.append((x, y))
        return valid_moves

    # 勝利パターンを検出（AIが一手で勝てるか）
    def detect_winning_move(self, player):
        valid_moves = self.get_valid_moves()
        for move in valid_moves:
            x, y = move
            # 一時的に石を置いてみる
            self.state[y][x] = player
            
            # 勝利条件チェック（5つ並ぶかどうか）
            if self.check_win(x, y, player):
                # 元に戻す
                self.state[y][x] = 0
                return move
            
            # 元に戻す
            self.state[y][x] = 0
        
        return None

    # 相手の勝利を阻止する手を検出
    def detect_blocking_move(self, player):
        # 相手のプレイヤー番号
        opponent = 1 if player == 2 else 2
        
        # 相手が次の手で勝てる場所を検出
        return self.detect_winning_move(opponent)

    # 勝利条件チェック
    def check_win(self, x, y, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、縦、右下がり斜め、右上がり斜め
        
        for dx, dy in directions:
            count = 1  # 自分自身
            
            # 正方向
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.state[ny][nx] == player:
                count += 1
                nx, ny = nx + dx, ny + dy
                
            # 逆方向
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.state[ny][nx] == player:
                count += 1
                nx, ny = nx - dx, ny - dy
                
            if count >= 5:
                return True
                
        return False

    # 相手の3つ並びを検出する関数
    def detect_three_in_a_row(self, player):
        opponent = 1 if player == 2 else 2
        candidate_moves = []
        
        # 方向ベクトル（横、縦、右下がり斜め、右上がり斜め）
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        # 盤面全体をチェック
        for y in range(self.board_size):
            for x in range(self.board_size):
                # 空白マスのみチェック
                if self.state[y][x] != 0:
                    continue
                
                # 各方向について3つ並びを検出
                for dx, dy in directions:
                    # パターン1: □●●●□ (両端が空いている場合のみ危険)
                    if (self.check_consecutive(x+dx, y+dy, dx, dy, opponent, 3) and
                        self.check_empty_position(x+dx*4, y+dy*4)):
                        candidate_moves.append((x, y))
                    
                    # パターン2: □●●●□ (反対側から見た場合)
                    if (self.check_consecutive(x-dx, y-dy, -dx, -dy, opponent, 3) and
                        self.check_empty_position(x-dx*4, y-dy*4)):
                        candidate_moves.append((x, y))
                    
                    # パターン3: □●□●●□ (間に1つ空きがある場合、両端が空いている場合のみ)
                    if (self.check_consecutive(x-dx, y-dy, -dx, -dy, opponent, 1) and 
                        self.check_consecutive(x+dx, y+dy, dx, dy, opponent, 2) and
                        self.check_empty_position(x-dx*2, y-dy*2) and
                        self.check_empty_position(x+dx*3, y+dy*3)):
                        candidate_moves.append((x, y))
                    
                    # パターン4: □●●□●□ (間に1つ空きがある別パターン、両端が空いている場合のみ)
                    if (self.check_consecutive(x-dx*2, y-dy*2, -dx, -dy, opponent, 2) and 
                        self.check_consecutive(x+dx, y+dy, dx, dy, opponent, 1) and
                        self.check_empty_position(x-dx*3, y-dy*3) and
                        self.check_empty_position(x+dx*2, y+dy*2)):
                        candidate_moves.append((x, y))
        
        return list(set(candidate_moves))  # 重複を除去
    
    # 指定方向に連続した石があるか確認
    def check_consecutive(self, x, y, dx, dy, player, count):
        for i in range(count):
            nx, ny = x + i*dx, y + i*dy
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size and self.state[ny][nx] == player):
                return False
        return True
    
    # 指定位置に指定プレイヤーの石があるか確認
    def check_position(self, x, y, player):
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return self.state[y][x] == player
        return False
    
    # 指定位置が空いているか確認
    def check_empty_position(self, x, y):
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return self.state[y][x] == 0
        return False

    # AIの手を決める（ルールベースとMCTSを組み合わせる）
    def get_ai_move(self):
        """AIの手を決める（定期的に画面を更新する）"""
        # AIプレイヤー番号を取得（環境の現在のプレイヤー）
        current_player = self.env.current_player
        start_time = pygame.time.get_ticks()
        
        # 定期的に画面を更新しながら処理
        def update_thinking_display():
            if pygame.time.get_ticks() - start_time > 100:  # 100msごとに更新
                self.draw_board()
                self.display_thinking(process=True)
        
        # 1. 勝利できる手があればそれを選択
        update_thinking_display()
        winning_move = self.detect_winning_move(current_player)
        if winning_move:
            print("AIが勝利パターンを検出しました")
            return winning_move
            
        # 2. 負けを回避する手があればそれを選択
        update_thinking_display()
        blocking_move = self.detect_blocking_move(current_player)
        if blocking_move:
            print("AIが防御パターンを検出しました")
            return blocking_move
        
        # 3. 相手の3つ並びを検出して対応
        update_thinking_display()
        three_in_row_moves = self.detect_three_in_a_row(current_player)
        if three_in_row_moves:
            print("AIが相手の3連を検出しました")
            
            # MCTSのポリシーを取得（処理中も画面更新）
            update_thinking_display()
            mcts_policy = self.mcts.search(np.array(self.state))
            update_thinking_display()
            
            # 候補手のうち、MCTSの評価が最も高いものを選択
            best_score = -1
            best_move = None
            for move in three_in_row_moves:
                x, y = move
                move_idx = y * self.board_size + x
                score = mcts_policy[move_idx]
                if score > best_score:
                    best_score = score
                    best_move = move
            
            if best_move:
                return best_move
        
        # 4. 上記に該当しない場合はMCTSで手を決める（長時間処理なので定期的に更新）
        update_thinking_display()
        mcts_policy = self.mcts.search(np.array(self.state))
        update_thinking_display()
        
        action_idx = np.argmax(mcts_policy)
        return (action_idx % self.board_size, action_idx // self.board_size)

    def get_human_action(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 左クリック
                    x, y = event.pos
                    board_offset_x = self.margin // 2
                    board_offset_y = 70
                    
                    # クリック位置から盤面の座標に変換（調整されたサイズに合わせて）
                    board_x = round((x - board_offset_x) / self.cell_size)
                    board_y = round((y - board_offset_y) / self.cell_size)
                    
                    if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                        if self.state[board_y][board_x] == 0:
                            return (board_x, board_y)

    # モデルのファイル名からボードサイズを抽出する関数
    def extract_board_size_from_model_path(self, model_path):
        try:
            # ファイル名からボードサイズを抽出（alpha_gomoku_8_iter...のような形式を想定）
            filename = os.path.basename(model_path)
            # アンダースコアで分割して、alpha_gomoku_の後の数字を取得
            parts = filename.split('_')
            if len(parts) >= 3 and parts[0] == 'alpha' and parts[1] == 'gomoku':
                try:
                    size = int(parts[2])
                    print(f"モデルから検出されたボードサイズ: {size}")
                    return size
                except ValueError:
                    pass
        except Exception as e:
            print(f"ボードサイズ検出エラー: {str(e)}")
        
        return self.board_size  # デフォルトのボードサイズを返す

    def display_thinking(self, process=False):
        """AIが思考中であることを表示"""
        thinking_overlay = pygame.Surface((300, 50), pygame.SRCALPHA)
        thinking_overlay.fill((0, 0, 0, 128))  # 半透明の黒背景
        
        font = self.get_jpn_font(24)
        
        # 処理中の場合はドットアニメーションを表示
        if process:
            # 現在の経過時間に基づいてドットの数を決める
            dots = "." * (int(pygame.time.get_ticks() / 500) % 4)
            thinking_text = font.render(f"AIが思考中{dots}", True, (255, 255, 255))
        else:
            thinking_text = font.render("AIが思考中...", True, (255, 255, 255))
            
        text_rect = thinking_text.get_rect(center=(150, 25))
        
        thinking_overlay.blit(thinking_text, text_rect)
        self.screen.blit(thinking_overlay, (self.screen_size//2 - 150, self.screen_size - 70))
        pygame.display.flip()  # 画面を更新
        
        # イベント処理（ウィンドウを閉じられるようにする）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def run(self):
        human_first = self.choose_player()
        self.human_is_black = human_first  # 人間が黒（先手）かどうかを保存
        done = False
        
        # モデル読み込みが失敗した場合、モデル名に基づいてボードサイズを調整して再試行
        if not hasattr(self.model, 'loaded_successfully'):
            detected_size = self.extract_board_size_from_model_path(self.model_path)
            if detected_size != self.board_size:
                print(f"検出されたボードサイズ({detected_size})に合わせてモデルを再読み込みします")
                self.board_size = detected_size
                self.screen_size = self.cell_size * self.board_size + 100
                # 画面サイズ再設定
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                # モデル再初期化
                self.model = DualNetwork(self.board_size).to(device)
                try:
                    self.model.load_state_dict(torch.load(self.model_path, map_location=device))
                    print(f"モデルを正常に読み込みました")
                    self.model.loaded_successfully = True
                except Exception as e:
                    print(f"モデルの再読み込みに失敗しました: {str(e)}")
                
                # 環境を再初期化
                self.env = GomokuEnv.GomokuEnv(board_size=self.board_size)
                self.state = self.env.board.GetBoardInt()
                
                # MCTSも再初期化
                self.mcts = MCTS(self.model, num_simulations=800)
        
        # 人間が後手の場合、AIが先に打つ
        if not human_first:
            self.draw_board()
            pygame.display.flip()
            
            # AIの思考中表示
            self.display_thinking()
            
            # AIの手番（ルールベース+MCTS）
            action = self.get_ai_move()
            next_state, reward, done, _ = self.env.step(action)
            self.state = next_state.cpu().numpy()
            self.last_move = action  # 最後の手を更新
            
            # AIが打った後に必ず画面を更新
            self.draw_board()
            pygame.display.flip()
        
        # メインゲームループ
        while not done:
            self.draw_board()
            pygame.display.flip()
            
            # 人間の手番
            action = self.get_human_action()
            next_state, reward, done, _ = self.env.step(action)
            self.state = next_state.cpu().numpy()
            self.last_move = action  # 最後の手を更新
            
            # 人間の手を表示
            self.draw_board()
            pygame.display.flip()
            
            if done:
                self.display_winner("あなたの勝ちです！")
                break
            
            # AIの思考中表示
            self.display_thinking()
            
            # AIの手番（ルールベース+MCTS）
            action = self.get_ai_move()
            next_state, reward, done, _ = self.env.step(action)
            self.state = next_state.cpu().numpy()
            self.last_move = action  # 最後の手を更新
            
            # AIの手を表示するために明示的に画面を更新
            self.draw_board()
            pygame.display.flip()
            
            if done:
                self.display_winner("AIの勝ちです!")
    
    def choose_player(self):
        """先手か後手かを選択するUI"""
        self.screen.fill((200, 200, 200))
        font = self.get_jpn_font(36)  # フォントサイズを大きく
        title = font.render("先手(黒)と後手(白)どちらにしますか？", True, (0, 0, 0))
        first_button = font.render("先手(黒)", True, (0, 0, 0))
        second_button = font.render("後手(白)", True, (0, 0, 0))
        
        title_rect = title.get_rect(center=(self.screen_size//2, self.screen_size//2 - 80))
        
        # ボタンのサイズも拡大
        first_rect = pygame.Rect(self.screen_size//2 - 150, self.screen_size//2, 120, 60)
        second_rect = pygame.Rect(self.screen_size//2 + 30, self.screen_size//2, 120, 60)
        
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
        font = self.get_jpn_font(40)  # フォントサイズを大きく
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
                    waiting = False  # キー入力またはマウスクリックでゲーム終了画面を閉じる

# 実行部分
gui = AlphaGomokuGUI(board_size=15, model_path='models/alpha_gomoku_15_iter235_trained_20250629_165719.pth')
gui.run()
