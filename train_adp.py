import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict
import math
import copy
import os
import matplotlib.pyplot as plt  # グラフ描画のためにmatplotlibをインポート

# Provided GomokuEnv classes (assuming they are in the same directory or accessible)
# from RandomGomoku.Board import Board
# from RandomGomoku.Dependency import Dependency
# from RandomGomoku.const import Stone

# Dummy classes for demonstration if the actual files are not available
# In a real scenario, you would use the actual imported classes.
class Board:
    def __init__(self):
        self.board = None
        self.width = 0
        self.height = 0

    def MakeBoard(self, width, height):
        self.width = width
        self.height = height
        self.board = [[0 for _ in range(width)] for _ in range(height)]

    def GetBoardInt(self):
        return self.board

    def SetBoard(self, board_state):
        self.board = [row[:] for row in board_state] # Deep copy

    def SetStone(self, x, y, stone):
        if not (0 <= x < self.width and 0 <= y < self.height and self.board[y][x] == 0):
            return False # Invalid move
        self.board[y][x] = stone.value # Assuming Stone has a .value attribute
        return self.CheckWin(stone) # Check for win after placing the stone

    def CheckWin(self, stone):
        # Simplistic win check for demonstration
        # In a real Gomoku game, this would be much more complex, checking all 8 directions
        # for 5 consecutive stones. This is a placeholder.
        s_val = stone.value
        
        # Check rows
        for r in range(self.height):
            for c in range(self.width - 4):
                if all(self.board[r][c+i] == s_val for i in range(5)):
                    return True
        # Check columns
        for c in range(self.width):
            for r in range(self.height - 4):
                if all(self.board[r+i][c] == s_val for i in range(5)):
                    return True
        # Check diagonals (top-left to bottom-right)
        for r in range(self.height - 4):
            for c in range(self.width - 4):
                if all(self.board[r+i][c+i] == s_val for i in range(5)):
                    return True
        # Check diagonals (top-right to bottom-left)
        for r in range(self.height - 4):
            for c in range(4, self.width):
                if all(self.board[r+i][c-i] == s_val for i in range(5)):
                    return True
        return False

    def PrintBoard(self):
        print(" " + " ".join([chr(ord('A') + i) for i in range(self.width)]))
        for r in range(self.height):
            row_str = str(r % 10) + " "
            for c in range(self.width):
                if self.board[r][c] == 0:
                    row_str += "+ "
                elif self.board[r][c] == 1:
                    row_str += "X "
                else:
                    row_str += "O "
            print(row_str)

class Dependency:
    def resolve(self, cls):
        return cls()

from enum import Enum
class Stone(Enum):
    NONE = 0
    BLACK = 1
    WHITE = 2

# Original GomokuEnv class
class GomokuEnv:
    def __init__(self, board_size=15, train_target="first", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.board_size = board_size
        self.container = Dependency()
        self.board: Board = self.container.resolve(Board)
        self.board.MakeBoard(board_size, board_size)
        self.stone = Stone.BLACK
        self.current_player = 1 # 1 for Black, 2 for White
        self.blackStones = 0
        self.whiteStones = 0
        self.device = device
        if train_target == "first":
            self.train_player = 1
        elif train_target == "second":
            self.train_player = 2
        else:
            raise ValueError("train_targetはfirstかsecondを指定してください")

    def reset(self):
        self.board.MakeBoard(self.board_size, self.board_size)
        self.stone = Stone.BLACK
        self.current_player = 1
        self.blackStones = 0
        self.whiteStones = 0
        board_tensor = torch.tensor(self.board.GetBoardInt(), dtype=torch.float32, device=self.device)
        return board_tensor

    def step(self, action):
        # Noneアクションのチェック
        if action is None:
            reward = torch.tensor(-10.0, device=self.device) # 無効な手は大きな負の報酬
            done = True
            board_tensor = torch.tensor(self.board.GetBoardInt(), dtype=torch.float32, device=self.device)
            return board_tensor, reward, done, {"invalid_action": True}
            
        x, y = action # action is (row, col) but Board.SetStone expects (col, row)
        
        # 範囲外のチェック
        if not (0 <= y < self.board_size and 0 <= x < self.board_size):
            reward = torch.tensor(-10.0, device=self.device)
            done = True
            board_tensor = torch.tensor(self.board.GetBoardInt(), dtype=torch.float32, device=self.device)
            return board_tensor, reward, done, {"invalid_action": True, "reason": "out_of_bounds"}

        if self.board.GetBoardInt()[y][x] != 0:
            # 無効なアクション：既に埋まっているセルが選択された
            reward = torch.tensor(-10.0, device=self.device) # 無効な手は大きな負の報酬
            done = True
            board_tensor = torch.tensor(self.board.GetBoardInt(), dtype=torch.float32, device=self.device)
            return board_tensor, reward, done, {"invalid_action": True, "reason": "occupied"}

        player_stone_enum = Stone.BLACK if self.current_player == 1 else Stone.WHITE
        
        if self.current_player == 1:
            self.blackStones += 1
        else:
            self.whiteStones += 1

        win_flag = self.board.SetStone(x, y, player_stone_enum) # Board.SetStone expects (col, row) for x,y

        # 石の数が正常かチェック (通常は交互に打つため、黒石と白石の差は0か1)
        if not (self.blackStones - self.whiteStones == 1 or self.blackStones == self.whiteStones):
            # ゲームの途中で石の数が不整合を起こすのは致命的なエラー
            raise ValueError("石の数がおかしいです")

        reward = torch.tensor(0.0, device=self.device)
        done = False

        if win_flag:
            done = True
            if self.current_player == self.train_player:
                reward = torch.tensor(1.0, device=self.device) # 訓練対象プレイヤーの勝利
            else:
                reward = torch.tensor(-1.0, device=self.device) # 訓練対象プレイヤーの敗北
        else:
            # 引き分け判定 (盤面が全て埋まった場合)
            if all(cell != 0 for row in self.board.GetBoardInt() for cell in row):
                done = True
                reward = torch.tensor(0.0, device=self.device) # 引き分け
        
        now_player = self.current_player
        self.current_player = 3 - self.current_player # プレイヤー交代

        board_tensor = torch.tensor(self.board.GetBoardInt(), dtype=torch.float32, device=self.device)
        return board_tensor, reward, done, {"which_player": now_player}

    def get_human_action(self):
        while True:
            try:
                x = int(input(f"x座標を入力してください (0-{self.board_size-1}): "))
                y = int(input(f"y座標を入力してください (0-{self.board_size-1}): "))
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board.GetBoardInt()[y][x] == 0:
                    break
                else:
                    print("そこには置けません。または範囲外です。")
            except ValueError:
                print("無効な入力です。数値を入力してください。")
        return (y, x) # Env's step expects (row, col)

    def render(self):
        self.board.PrintBoard()

    def get_legal_moves(self):
        moves = []
        board_state = self.board.GetBoardInt()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == 0:
                    moves.append((r, c))
        random.shuffle(moves)  # ランダムに合法手をシャッフル
        return moves


class ADPNetwork(nn.Module):
    def __init__(self, board_size=15, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(ADPNetwork, self).__init__()
        self.board_size = board_size
        self.device = device

        # 論文に記載の入力層のノード数274、隠れ層100、出力層1 [cite: 110, 346]
        # 入力層の具体的なマッピングはTable Iに基づいているが、ここでは汎用的な盤面表現を仮定
        # 実際には、特定のパターン（例：4連、3連など）を検出して入力に変換する必要がある [cite: 111, 347]
        # ここでは、簡略化のため、各セルの状態（空、黒、白）を直接one-hotエンコーディングに似た形で表現する。
        # 各セルは黒か白のどちらかの石が置かれるため、2 * board_size^2 の特徴量で表現
        self.input_size = board_size * board_size * 2 
        self.hidden_size = 100  # [cite: 110, 346]
        self.output_size = 1  # [cite: 110, 346] # 勝率

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)  # バッチ正規化を追加
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid() # 勝率なので0-1の範囲に収める [cite: 125, 361]
        
        # 重みの初期化
        self.reset_parameters()
        
        self.to(self.device)
    
    def reset_parameters(self):
        # He初期化を使用して勾配消失/爆発問題を軽減
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        # x: (バッチサイズ, board_size * board_size * 2)
        x = torch.relu(self.bn1(self.fc1(x)))  # バッチ正規化を適用
        x = self.sigmoid(self.fc2(x))
        return x

    def get_board_features(self, board_state, current_player):
        # 盤面状態をニューラルネットワークの入力特徴量に変換
        # 論文のTable Iに示された「コーディング方法」を簡易的に実装 [cite: 113, 349]
        # 実際には、五目並べの特定のパターン（オープンエンドの3連、4連など）を検出して、
        # それらの数を特徴量として入力にエンコードする必要がある [cite: 111, 347]。
        
        # 現在のプレイヤーが黒(1)の場合、features[r, c, 0]が黒石、features[r, c, 1]が白石
        # 現在のプレイヤーが白(2)の場合、features[r, c, 0]が白石、features[r, c, 1]が黒石
        # このようにすることで、ネットワークは常に「自分の石」と「相手の石」を区別できる
        features = np.zeros((self.board_size, self.board_size, 2), dtype=np.float32)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == current_player: # 自分の石
                    features[r, c, 0] = 1
                elif board_state[r][c] == (3 - current_player): # 相手の石
                    features[r, c, 1] = 1
        return torch.from_numpy(features.flatten()).float().to(self.device)

    def predict_win_probability(self, board_state, current_player):
        self.eval() # 評価モード
        with torch.no_grad():
            features = self.get_board_features(board_state, current_player).unsqueeze(0) # バッチ次元を追加
            win_prob = self.forward(features).item()
        return win_prob

    def get_candidate_moves(self, board_state, current_player, num_candidates=5):
        # ADPネットワークを使って候補手を生成 [cite: 23, 259]
        # 論文では、ADPによって訓練されたNNからトップ5の勝率を持つ候補手を得る [cite: 118, 354]。
        
        legal_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == 0:
                    legal_moves.append((r, c))

        if not legal_moves:
            return [], []

        move_evaluations = []
        for move in legal_moves:
            temp_board = [row[:] for row in board_state]
            # 各合法手を置いたと仮定した新しい盤面状態を生成し、その状態の勝率を予測
            temp_board[move[0]][move[1]] = current_player
            prob = self.predict_win_probability(temp_board, current_player)
            move_evaluations.append((move, prob))
        random.shuffle(move_evaluations)  # 評価の順序をランダム化して探索の多様性を確保
        # 勝率の高い順にソート
        move_evaluations.sort(key=lambda x: x[1], reverse=True)
        
        # トップNの候補手とその勝率を取得
        candidate_moves = [move for move, prob in move_evaluations[:num_candidates]]
        adp_win_probabilities = [prob for move, prob in move_evaluations[:num_candidates]]
        
        return candidate_moves, adp_win_probabilities


class MCTSNode:
    def __init__(self, board_state, current_player, parent=None, move=None):
        self.board_state = copy.deepcopy(board_state)
        self.parent = parent
        self.move = move  # このノードに至った手 (row, col)
        self.current_player = current_player  # このノードの状態での次の手番のプレイヤー (石を置くプレイヤー)

        self.children = {}  # 手 -> 子ノードのマッピング
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = self._get_legal_moves(board_state) # 試行されていない合法手

    def _get_legal_moves(self, board_state):
        moves = []
        for r in range(len(board_state)):
            for c in range(len(board_state[0])):
                if board_state[r][c] == 0:
                    moves.append((r, c))
        random.shuffle(moves)
        return moves

    def is_terminal(self):
        # 盤面が終了状態（勝敗が決まったか、引き分け）であるかをチェック
        temp_board_obj = Board()
        temp_board_obj.MakeBoard(len(self.board_state), len(self.board_state[0]))
        temp_board_obj.SetBoard(self.board_state)
        
        # 勝利プレイヤーが存在するか
        if temp_board_obj.CheckWin(Stone.BLACK) or temp_board_obj.CheckWin(Stone.WHITE):
            return True
        
        # 全てのマスが埋まっているか (引き分け)
        if all(cell != 0 for row in self.board_state for cell in row):
            return True
            
        return False

    def get_reward(self, winner_player):
        # 終端状態からの報酬を決定 [cite: 79, 315]
        # Gomokuは通常引き分けが稀なゼロサムゲーム [cite: 77, 313, 78, 314]
        # 勝った場合は1、負けまたは引き分けは0 [cite: 79, 315]
        
        # `winner_player` はゲームを終了させたプレイヤー。
        # ここでは、MCTSのシミュレーションの報酬をMCTSの探索を開始したルートノードのプレイヤー視点で考える。
        # つまり、ルートノードのプレイヤーが勝てば1、負ければ0。
        
        # ただし、MCTSノードの報酬は通常、そのノードの`current_player`視点で考えるべき。
        # Simulation関数が、シミュレーションを開始したプレイヤーの勝利を1、敗北を0と返す場合、
        # Backpropagationでは、親ノードの視点に合わせて報酬を反転させる必要がある。
        # ここでは、シミュレーションの戻り値を「シミュレーション開始プレイヤーが勝つ確率」と仮定し、
        # Backpropagationで適切に報酬を調整する。

        temp_board_obj = Board()
        temp_board_obj.MakeBoard(len(self.board_state), len(self.board_state[0]))
        temp_board_obj.SetBoard(self.board_state)

        if temp_board_obj.CheckWin(Stone.BLACK):
            actual_winner = 1
        elif temp_board_obj.CheckWin(Stone.WHITE):
            actual_winner = 2
        else: # 引き分け
            actual_winner = 0

        # MCTSノードのcurrent_playerが勝った場合を1、そうでない場合を0とする。
        if actual_winner == 0: # 引き分け
            return 0.0
        return 1.0 if actual_winner == self.current_player else 0.0

    def expand(self):
        # 未試行の合法手から一つ選んで新しい子ノードを作成 [cite: 62, 298]
        move = self.untried_moves.pop()
        new_board_state = copy.deepcopy(self.board_state)
        
        # 現在のプレイヤーの石を置く
        new_board_state[move[0]][move[1]] = self.current_player
        
        # 次のプレイヤーは3 - self.current_player
        next_player = 3 - self.current_player
        child_node = MCTSNode(new_board_state, next_player, parent=self, move=move)
        self.children[move] = child_node
        return child_node

    def best_child(self, c_param=math.sqrt(2)): # 論文のAlgorithm 2のBest Child関数における 'c' パラメータに相当
        # UCB1式に基づいて最適な子ノードを選択 [cite: 85, 321]
        # UCB = x_j_bar + sqrt(2 * ln(n) / n_j) [cite: 85, 321]
        # x_j_bar: 平均報酬 (wins / visits)
        # n: 親ノードの訪問回数 (self.visits)
        # n_j: 子ノードの訪問回数 (child.visits)
        
        def uct_value(child):
            if child.visits == 0:
                # 訪問回数が0の子ノードは無限大のUCB値を持つとみなし、優先的に探索する
                return float('inf')
            
            # 平均報酬 (exploitation term)
            # `wins`は、そのノードの視点での勝利数。
            # MCTSのSimulationは、シミュレーションを開始したノードの`current_player`視点での勝敗を返す。
            # Backpropagationで親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換されるべき。
            # たとえば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
            # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
            exploitation_term = child.wins / child.visits
            
            # 探索項 (exploration term)
            exploration_term = c_param * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation_term + exploration_term

        # 全ての子ノードのUCB値を計算し、最大値を持つ子ノードを返す
        return max(self.children.values(), key=uct_value)


class MCTS:
    def __init__(self, board_size=15, simulation_times=400, heuristic_knowledge_enabled=True):
        self.board_size = board_size
        self.simulation_times = simulation_times
        self.heuristic_knowledge_enabled = heuristic_knowledge_enabled

    def _apply_heuristic_knowledge(self, board_state, current_player):
        # 論文のAlgorithm 1とAlgorithm 2で述べられているヒューリスティック知識を適用 [cite: 68, 304, 88, 324]
        # 具体的なルール: [cite: 71, 307]
        # 1. 自分の4連がある場合、5連になる位置に強制的に移動 [cite: 71, 307]
        # 2. 相手の4連がある場合、5連をブロックする位置に強制的に移動 [cite:  72, 308]
        # 3. 自分の3連がある場合、4連になる位置に強制的に移動 [cite: 73, 309]
        # 4. 相手の3連がある場合、4連をブロックする位置に強制的に移動 [cite: 76, 312]
        
        # このメソッドは、最も優先度の高い強制的な手があればそれを返し、なければNoneを返す。
        
        temp_board_obj = Board()
        temp_board_obj.MakeBoard(self.board_size, self.board_size)
        
        legal_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == 0:
                    legal_moves.append((r, c))
        random.shuffle(legal_moves)  # ランダムに合法手をシャッフル

        # 優先度1: 自分の5連を完成させる手
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(current_player)):
                return (r, c)
        
        # 優先度2: 相手の5連をブロックする手
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(opponent_player)):
                return (r, c)
        
        # 優先度3: 自分の3連から4連を作る手 [cite: 73, 309]
        # 各方向（水平、垂直、斜め）で3連のパターンを探す
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連から4連になるか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 4連が形成される場合、このセルを優先
                if count == 4:
                    # 端点が空いているか確認（オープン4の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        # 優先度4: 相手の3連をブロックする手 [cite: 76, 312]
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            # まず相手がこの位置に石を置いたと仮定
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連を形成するか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 相手が3連または4連を形成する場合、このセルをブロックする
                if count >= 3:
                    # 端点が空いているか確認（オープン3の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        return None # 強制手が見つからなかった場合

    def _simulate_game(self, board_state, starting_player):
        # シミュレーションステージ: リーフノードからゲームを終了までランダムにプレイアウトする [cite: 63, 299]
        # 論文のSimulation関数に相当 [cite: 68, 304]
        
        current_board = copy.deepcopy(board_state)
        current_player = starting_player
        
        temp_env = GomokuEnv(board_size=self.board_size)
        
        while True:
            temp_env.board.SetBoard(current_board)
            
            # 勝利判定 [cite: 68, 304]
            if temp_env.board.CheckWin(Stone.BLACK):
                return 1.0 if starting_player == 1 else 0.0 # 黒が勝ち、報酬を調整
            if temp_env.board.CheckWin(Stone.WHITE):
                return 1.0 if starting_player == 2 else 0.0 # 白が勝ち、報酬を調整
            
            legal_moves = []
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if current_board[r][c] == 0:
                        legal_moves.append((r, c))
            random.shuffle(legal_moves)  # ランダムに合法手をシャッフル

            if not legal_moves: # 引き分け [cite: 79, 315]
                return 0.0 # 引き分けは報酬0 [cite: 79, 315]

            chosen_move = None
            if self.heuristic_knowledge_enabled:
                # 強制手があればそれを選択 [cite: 68, 304]
                forced_action = self._apply_heuristic_knowledge(current_board, current_player)
                if forced_action and forced_action in legal_moves: # 強制手が合法手であることも確認
                    chosen_move = forced_action
            
            if chosen_move is None:
                chosen_move = random.choice(legal_moves) # 強制手がない場合、ランダムに選択 [cite: 68, 304]
            
            current_board[chosen_move[0]][chosen_move[1]] = current_player
            current_player = 3 - current_player # プレイヤー交代

    def _backpropagate(self, node, reward):
        # 逆伝播ステージ: シミュレーション結果を選択されたノードに逆伝播させ、状態値を更新する [cite: 64, 300]
        # 論文のBack Update関数に相当 [cite: 92, 328]
        
        while node is not None:
            node.visits += 1  # [cite: 92, 328]
            
            # 報酬の調整：親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換する必要がある。
            # 例えば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
            # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
            if node.current_player == (3 - (node.parent.current_player if node.parent else starting_player)):
                # ノードが親から遷移してきた手番のプレイヤー視点の場合
                # Simulatrionの報酬は、シミュレーションを開始したノードの`current_player`視点での勝敗を返す。
                # Backpropagationで親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換されるべき。
                # たとえば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
                # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
                node.wins += reward
            else:
                # 相手の視点の場合
                node.wins += (1.0 - reward)
            
            node = node.parent

    def run_mcts(self, root_board_state, root_player):
        # MCTSの主要な処理を統合
        # root_board_state: MCTSのルートノードとなる盤面状態
        # root_player: ルートノードでの手番のプレイヤー
        
        root_node = MCTSNode(root_board_state, current_player=root_player)

        for _ in range(self.simulation_times):
            # 1. 選択 (Selection) [cite: 60, 296, 61, 297]
            node = root_node
            # 全ての子ノードが展開されているか、終端ノードに到達するまで深く潜る
            while node.untried_moves == [] and node.children != {}:
                if node.is_terminal(): # 終端ノードに到達したらループを抜ける
                    break
                node = node.best_child()
            
            # 2. 展開 (Expansion) [cite: 62, 298]
            # 全ての子ノードが展開されていない場合、未試行の合法手から一つ選んで新しい子ノードを作成
            if node.untried_moves != [] and not node.is_terminal():
                node = node.expand()

            # 3. シミュレーション (Simulation) [cite: 63, 299]
            # リーフノードからゲームを終了までランダムにプレイアウトする
            # シミュレーションは展開されたノードから開始される
            # ここでのwinnerは「シミュレーションを開始したノードのプレイヤー」が勝った場合1、負けた場合0
            winner_reward = self._simulate_game(node.board_state, node.current_player)
            
            # 4. 逆伝播 (Backpropagation) [cite: 64, 300]
            # シミュレーション結果を選択されたノードに逆伝播させ、状態値を更新する
            self._backpropagate(node, winner_reward)

        # MCTSが終了したら、最も訪問回数が多い（または勝率が高い）ルートの子ノードに対応する手を返す
        # 論文のAlgorithm 2のBestChild(v0) [cite: 92, 328]
        if not root_node.children:
            return 0.0 # 候補手がない場合 (ゲームが既に終了している場合など)
            
        # 論文では、MCTSの勝率は wins / visits で表される [cite: 85, 321]。
        # ここでは、ルートノードの子ノード（各候補手）の勝率のうち、最も良いものを返す。
        # または、最終的に選択されるアクションのMCTS勝率を返す。
        
        # MCTSの勝率は、そのアクションを選択したルートノードの子ノードの`wins / visits`である。
        # 最終的にADPとMCTSを統合して手を選ぶため、ここでは各候補手に対するMCTS勝率を返すのではなく、
        # `ADPMCTS.select_action`で各候補手に対応するMCTS勝率を計算させる。
        # ここでは、指定されたルートからのMCTSの勝率として、ベストな子ノードの勝率を返す。
        
        # MCTSStageでは、各ADP候補手mに対して`MTCS(m,s)`から`w2`を得るとある [cite: 137, 373]。
        # これは、各ADP候補手を置いた後の盤面をMCTSのルートとして探索し、その探索結果としての勝率を意味する。
        # よって、ここではMCTSそのものの評価値を返す。
        
        # ここでは、ルートノードの各子ノード（つまり、可能な次の手）について、そのMCTSでの勝率を計算して返す
        mcts_win_probabilities = {}
        for move, child_node in root_node.children.items():
            if child_node.visits > 0:
                mcts_win_probabilities[move] = child_node.wins / child_node.visits
            else:
                mcts_win_probabilities[move] = 0.0 # 訪問されていないノードは勝率0

        return mcts_win_probabilities # 各候補手に対応するMCTS勝率の辞書


class ADPMCTS:
    def __init__(self, board_size=15, adp_model_path=None, mcts_simulation_times=400, lambda_param=0.5, num_adp_candidates=5):
        self.board_size = board_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adp_network = ADPNetwork(board_size=board_size, device=self.device)
        
        if adp_model_path:
            try:
                self.adp_network.load_state_dict(torch.load(adp_model_path, map_location=self.device))
                self.adp_network.eval() # 評価モードに設定
                print(f"ADPモデルを {adp_model_path} からロードしました。")
            except FileNotFoundError:
                print(f"警告: {adp_model_path} が見つかりませんでした。ADPモデルはランダムに初期化されます。")
            except Exception as e:
                print(f"警告: ADPモデルのロード中にエラーが発生しました: {e}。モデルはランダムに初期化されます。")
        else:
            print("警告: ADPモデルのパスが指定されていません。モデルはランダムに初期化されます。")

        self.mcts = MCTS(board_size=board_size, simulation_times=mcts_simulation_times)
        self.lambda_param = lambda_param # 論文のλパラメータ [cite: 128, 364]
        self.num_adp_candidates = num_adp_candidates # ADPから取得する候補手の数 [cite: 142, 378]
        
    def select_action(self, current_board_state, current_player):
        # 1. ADPステージ: ADPネットワークから上位5つの候補手とそのADP勝率を取得 [cite: 120, 356]
        # ADPStage(s0) [cite: 132, 368]
        adp_candidate_moves, adp_win_probabilities = self.adp_network.get_candidate_moves(
            current_board_state, current_player, self.num_adp_candidates
        )
        
        if not adp_candidate_moves:
            return None # 合法手がない場合

        final_predictions = []
        for i, move in enumerate(adp_candidate_moves):
            # 2. MCTSステージ: 各候補手に対してMCTSを実行し、MCTS勝率を取得 [cite: 122, 358]
            # MCTSStage(MADP) [cite: 132, 368]
            
            temp_board_for_mcts = copy.deepcopy(current_board_state)
            temp_board_for_mcts[move[0]][move[1]] = current_player
            
            # MCTSは、候補手を置いた後の盤面 (`temp_board_for_mcts`) をルートノードとし、
            # その盤面で手番を持つプレイヤー (`next_player_for_mcts`) の視点での勝率を計算する。
            next_player_for_mcts = 3 - current_player 
            
            # MCTS.run_mcts は、各合法手に対するMCTS勝率の辞書を返す。
            # ここでは、`move` (ADP候補手によって遷移した盤面) からのMCTS勝率を知りたい。
            # MCTSのルートノードが`temp_board_for_mcts`の場合、そのルートノードにおける勝率を評価する。
            
            # MCTS.run_mcts は、ルートノードの子ノードの勝率辞書を返すべきだが、
            # 論文の`MTCS(m,s)`は単一の勝率`w2`を返すことを示唆している [cite: 137, 373]。
            # ここでは、MCTSを一度実行して、そのルートノードの評価値（勝率）を得ると解釈する。
            # MCTSの評価は通常、ルートノードを十分に探索した後の、そのルートノード自体の評価である。
            # そのため、MCTSクラスには、ルートノードの勝率を計算するメソッドを追加する。
            # ただし、現状のMCTS.run_mctsは子ノードの勝率辞書を返すので、
            # その辞書から、そのMCTSのルートノード自体の勝率を評価するような適切な値を取得する必要がある。
            # 最も一般的なMCTSの評価は、ルートノードの最も有望な子ノードの勝率、またはルートノード自体のQ値。
            # 論文の`MTCS(m,s)`の出力`w2`は、MCTSシミュレーションによって得られた「MCTS winning probability」 [cite: 241, 25]
            # を指すので、そのMCTSの探索結果として、そのルートノードがどれだけ有望かという評価値を得る。
            # ここでは、MCTSが返す辞書の中から、最も勝率が高いと判断された手の勝率、または特定の評価値をMCTS勝率とする。
            # 簡潔化のため、MCTS.run_mctsは、特定の候補手に対するMCTSの勝率を直接返すものと仮定し、
            # MCTS内部でベストな子ノードの勝率を返すように修正する。
            
            # MCTS.run_mctsを呼び出す前に、`MCTS`クラスの`run_mcts`メソッドが
            # MCTSルートノード自体の勝率（そのノードの wins / visits）を返すように変更が必要。
            # または、`select_action`内でMCTSの根となるノードの子ノードの勝率を考慮する。
            
            # MCTSの`run_mcts`は、各候補手（ADPによって選択された手）に対応する盤面をMCTSのルートとして探索し、
            # そのMCTSの結果としての「勝率」を返すものとする。
            # MCTSの勝率は、そのMCTSの探索を完了した後の、ルートノード（ADPの候補手によって遷移した盤面）の評価値。
            # MCTSの評価は、そのルートノードの子ノードのQ値の最大値などで行われる。
            # 論文の`MTCS(m,s)`の出力`w2`は、MCTSシミュレーションによって得られた「MCTS winning probability」 [cite: 241, 25]
            # を指すので、そのMCTSの探索結果として、そのルートノードがどれだけ有望かという評価値を得る。
            # ここでは、MCTSが返す辞書の中から、最も勝率が高いと判断された手の勝率、または特定の評価値をMCTS勝率とする。
            # 簡潔化のため、MCTS.run_mctsは、特定の候補手に対するMCTSの勝率を直接返すものと仮定し、
            # MCTS内部でベストな子ノードの勝率を返すように修正する。
            
            mcts_win_prob = self.mcts.run_mcts_single_eval(temp_board_for_mcts, next_player_for_mcts)

            # 3. 最終予測勝率の計算: ADP勝率とMCTS勝率を重み付けして結合 [cite: 123, 359]
            # 論文の式(7): wp = λ * w1 + (1 - λ) * w2 [cite: 128, 364]
            # w1: ADPの勝率 [cite: 128, 364]
            # w2: MCTSの勝率 [cite: 128, 364]
            
            # MCTSの勝率 `mcts_win_prob` は、`next_player_for_mcts` が勝つ確率。
            # ADPの勝率 `adp_win_probabilities[i]` は、`current_player` が勝つ確率。
            # したがって、`mcts_win_prob` を `current_player` 視点に合わせるには `1 - mcts_win_prob` とすべき。
            # 論文の記述「final win rate both from ADP and MCTS algorithms」 [cite: 55, 291]
            # 「winRate against ADP depends on λ」 [cite: 144, 380] から、
            # どちらも「そのプレイヤーが勝つ確率」として扱っていると解釈するのが自然。
            # そのため、MCTSのシミュレーション報酬を「MCTSのルートノードのプレイヤーにとっての報酬」として定義する。
            # そうすることで、`mcts_win_prob` は ADPの勝率と同様に「現在のプレイヤーが勝つ確率」となる。

            combined_win_prob = self.lambda_param * adp_win_probabilities[i] + \
                                (1 - self.lambda_param) * mcts_win_prob
            
            final_predictions.append((move, combined_win_prob))
            
        # 4. 最も高い最終予測勝率を持つ手を選択 [cite: 26, 262]
        if not final_predictions:
            return None # 候補手がない場合

        best_move, _ = max(final_predictions, key=lambda x: x[1])
        return best_move

# MCTSクラスのrun_mctsを修正し、MCTSStageの挙動に合わせる
class MCTS:
    def __init__(self, board_size=15, simulation_times=400, heuristic_knowledge_enabled=True):
        self.board_size = board_size
        self.simulation_times = simulation_times
        self.heuristic_knowledge_enabled = heuristic_knowledge_enabled

    def _apply_heuristic_knowledge(self, board_state, current_player):
        # 論文のAlgorithm 1とAlgorithm 2で述べられているヒューリスティック知識を適用 [cite: 68, 304, 88, 324]
        # 具体的なルール: [cite: 71, 307]
        # 1. 自分の4連がある場合、5連になる位置に強制的に移動 [cite: 71, 307]
        # 2. 相手の4連がある場合、5連をブロックする位置に強制的に移動 [cite: 72, 308]
        # 3. 自分の3連がある場合、4連になる位置に強制的に移動 [cite: 73, 309]
        # 4. 相手の3連がある場合、4連をブロックする位置に強制的に移動 [cite: 76, 312]
        
        # このメソッドは、最も優先度の高い強制的な手があればそれを返し、なければNoneを返す。
        
        temp_board_obj = Board()
        temp_board_obj.MakeBoard(self.board_size, self.board_size)
        
        legal_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == 0:
                    legal_moves.append((r, c))
        random.shuffle(legal_moves)  # ランダムに合法手をシャッフル

        # 優先度1: 自分の5連を完成させる手
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(current_player)):
                return (r, c)
        
        # 優先度2: 相手の5連をブロックする手
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(opponent_player)):
                return (r, c)
        
        # 優先度3: 自分の3連から4連を作る手 [cite: 73, 309]
        # 各方向（水平、垂直、斜め）で3連のパターンを探す
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連から4連になるか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 4連が形成される場合、このセルを優先
                if count == 4:
                    # 端点が空いているか確認（オープン4の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        # 優先度4: 相手の3連をブロックする手 [cite: 76, 312]
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            # まず相手がこの位置に石を置いたと仮定
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連を形成するか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 相手が3連または4連を形成する場合、このセルをブロックする
                if count >= 3:
                    # 端点が空いているか確認（オープン3の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        return None # 強制手が見つからなかった場合

    def _simulate_game(self, board_state, starting_player):
        # シミュレーションステージ: リーフノードからゲームを終了までランダムにプレイアウトする [cite: 63, 299]
        # 論文のSimulation関数に相当 [cite: 68, 304]
        
        current_board = copy.deepcopy(board_state)
        current_player = starting_player
        
        temp_env = GomokuEnv(board_size=self.board_size)
        
        while True:
            temp_env.board.SetBoard(current_board)
            
            # 勝利判定 [cite: 68, 304]
            if temp_env.board.CheckWin(Stone.BLACK):
                return 1.0 if starting_player == 1 else 0.0 # 黒が勝ち、報酬を調整
            if temp_env.board.CheckWin(Stone.WHITE):
                return 1.0 if starting_player == 2 else 0.0 # 白が勝ち、報酬を調整
            
            legal_moves = []
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if current_board[r][c] == 0:
                        legal_moves.append((r, c))
            random.shuffle(legal_moves)  # ランダムに合法手をシャッフル

            if not legal_moves: # 引き分け [cite: 79, 315]
                return 0.0 # 引き分けは報酬0 [cite: 79, 315]

            chosen_move = None
            if self.heuristic_knowledge_enabled:
                # 強制手があればそれを選択 [cite: 68, 304]
                forced_action = self._apply_heuristic_knowledge(current_board, current_player)
                if forced_action and forced_action in legal_moves: # 強制手が合法手であることも確認
                    chosen_move = forced_action
            
            if chosen_move is None:
                chosen_move = random.choice(legal_moves) # 強制手がない場合、ランダムに選択 [cite: 68, 304]
            
            current_board[chosen_move[0]][chosen_move[1]] = current_player
            current_player = 3 - current_player # プレイヤー交代

    def _backpropagate(self, node, reward):
        # 逆伝播ステージ: シミュレーション結果を選択されたノードに逆伝播させ、状態値を更新する [cite: 64, 300]
        # 論文のBack Update関数に相当 [cite: 92, 328]
        
        while node is not None:
            node.visits += 1  # [cite: 92, 328]
            
            # 報酬の調整：親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換する必要がある。
            # 例えば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
            # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
            if node.current_player == (3 - (node.parent.current_player if node.parent else starting_player)):
                # ノードが親から遷移してきた手番のプレイヤー視点の場合
                # Simulatrionの報酬は、シミュレーションを開始したノードの`current_player`視点での勝敗を返す。
                # Backpropagationで親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換されるべき。
                # たとえば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
                # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
                node.wins += reward
            else:
                # 相手の視点の場合
                node.wins += (1.0 - reward)
            
            node = node.parent

    def run_mcts(self, root_board_state, root_player):
        # MCTSの主要な処理を統合
        # root_board_state: MCTSのルートノードとなる盤面状態
        # root_player: ルートノードでの手番のプレイヤー
        
        root_node = MCTSNode(root_board_state, current_player=root_player)

        for _ in range(self.simulation_times):
            # 1. 選択 (Selection) [cite: 60, 296, 61, 297]
            node = root_node
            # 全ての子ノードが展開されているか、終端ノードに到達するまで深く潜る
            while node.untried_moves == [] and node.children != {}:
                if node.is_terminal(): # 終端ノードに到達したらループを抜ける
                    break
                node = node.best_child()
            
            # 2. 展開 (Expansion) [cite: 62, 298]
            # 全ての子ノードが展開されていない場合、未試行の合法手から一つ選んで新しい子ノードを作成
            if node.untried_moves != [] and not node.is_terminal():
                node = node.expand()

            # 3. シミュレーション (Simulation) [cite: 63, 299]
            # リーフノードからゲームを終了までランダムにプレイアウトする
            # シミュレーションは展開されたノードから開始される
            # ここでのwinnerは「シミュレーションを開始したノードのプレイヤー」が勝った場合1、負けた場合0
            winner_reward = self._simulate_game(node.board_state, node.current_player)
            
            # 4. 逆伝播 (Backpropagation) [cite: 64, 300]
            # シミュレーション結果を選択されたノードに逆伝播させ、状態値を更新する
            self._backpropagate(node, winner_reward)

        # MCTSが終了したら、最も訪問回数が多い（または勝率が高い）ルートの子ノードに対応する手を返す
        # 論文のAlgorithm 2のBestChild(v0) [cite: 92, 328]
        if not root_node.children:
            return 0.0 # 候補手がない場合 (ゲームが既に終了している場合など)
            
        # 論文では、MCTSの勝率は wins / visits で表される [cite: 85, 321]。
        # ここでは、ルートノードの子ノード（各候補手）の勝率のうち、最も良いものを返す。
        # または、最終的に選択されるアクションのMCTS勝率を返す。
        
        # MCTSの勝率は、そのアクションを選択したルートノードの子ノードの`wins / visits`である。
        # 最終的にADPとMCTSを統合して手を選ぶため、ここでは各候補手に対するMCTS勝率を返すのではなく、
        # `ADPMCTS.select_action`で各候補手に対応するMCTS勝率を計算させる。
        # ここでは、指定されたルートからのMCTSの勝率として、ベストな子ノードの勝率を返す。
        
        # MCTSStageでは、各ADP候補手mに対して`MTCS(m,s)`から`w2`を得るとある [cite: 137, 373]。
        # これは、各ADP候補手を置いた後の盤面をMCTSのルートとして探索し、その探索結果としての勝率を意味する。
        # よって、ここではMCTSそのものの評価値を返す。
        
        # ここでは、ルートノードの各子ノード（つまり、可能な次の手）について、そのMCTSでの勝率を計算して返す
        mcts_win_probabilities = {}
        for move, child_node in root_node.children.items():
            if child_node.visits > 0:
                mcts_win_probabilities[move] = child_node.wins / child_node.visits
            else:
                mcts_win_probabilities[move] = 0.0 # 訪問されていないノードは勝率0

        return mcts_win_probabilities # 各候補手に対応するMCTS勝率の辞書


class ADPMCTS:
    def __init__(self, board_size=15, adp_model_path=None, mcts_simulation_times=400, lambda_param=0.5, num_adp_candidates=5):
        self.board_size = board_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adp_network = ADPNetwork(board_size=board_size, device=self.device)
        
        if adp_model_path:
            try:
                self.adp_network.load_state_dict(torch.load(adp_model_path, map_location=self.device))
                self.adp_network.eval() # 評価モードに設定
                print(f"ADPモデルを {adp_model_path} からロードしました。")
            except FileNotFoundError:
                print(f"警告: {adp_model_path} が見つかりませんでした。ADPモデルはランダムに初期化されます。")
            except Exception as e:
                print(f"警告: ADPモデルのロード中にエラーが発生しました: {e}。モデルはランダムに初期化されます。")
        else:
            print("警告: ADPモデルのパスが指定されていません。モデルはランダムに初期化されます。")

        self.mcts = MCTS(board_size=board_size, simulation_times=mcts_simulation_times)
        self.lambda_param = lambda_param # 論文のλパラメータ [cite: 128, 364]
        self.num_adp_candidates = num_adp_candidates # ADPから取得する候補手の数 [cite: 142, 378]
        
    def select_action(self, current_board_state, current_player):
        # 1. ADPステージ: ADPネットワークから上位5つの候補手とそのADP勝率を取得 [cite: 120, 356]
        # ADPStage(s0) [cite: 132, 368]
        adp_candidate_moves, adp_win_probabilities = self.adp_network.get_candidate_moves(
            current_board_state, current_player, self.num_adp_candidates
        )
        
        if not adp_candidate_moves:
            return None # 合法手がない場合

        final_predictions = []
        for i, move in enumerate(adp_candidate_moves):
            # 2. MCTSステージ: 各候補手に対してMCTSを実行し、MCTS勝率を取得 [cite: 122, 358]
            # MCTSStage(MADP) [cite: 132, 368]
            
            temp_board_for_mcts = copy.deepcopy(current_board_state)
            temp_board_for_mcts[move[0]][move[1]] = current_player
            
            # MCTSは、候補手を置いた後の盤面 (`temp_board_for_mcts`) をルートノードとし、
            # その盤面で手番を持つプレイヤー (`next_player_for_mcts`) の視点での勝率を計算する。
            next_player_for_mcts = 3 - current_player 
            
            # MCTS.run_mcts は、各合法手に対するMCTS勝率の辞書を返す。
            # ここでは、`move` (ADP候補手によって遷移した盤面) からのMCTS勝率を知りたい。
            # MCTSのルートノードが`temp_board_for_mcts`の場合、そのルートノードにおける勝率を評価する。
            
            # MCTS.run_mcts は、ルートノードの子ノードの勝率辞書を返すべきだが、
            # 論文の`MTCS(m,s)`は単一の勝率`w2`を返すことを示唆している [cite: 137, 373]。
            # ここでは、MCTSを一度実行して、そのルートノードの評価値（勝率）を得ると解釈する。
            # MCTSの評価は通常、ルートノードを十分に探索した後の、そのルートノード自体の評価である。
            # そのため、MCTSクラスには、ルートノードの勝率を計算するメソッドを追加する。
            # ただし、現状のMCTS.run_mctsは子ノードの勝率辞書を返すので、
            # その辞書から、そのMCTSのルートノード自体の勝率を評価するような適切な値を取得する必要がある。
            # 最も一般的なMCTSの評価は、ルートノードの最も有望な子ノードの勝率、またはルートノード自体のQ値。
            # 論文の`MTCS(m,s)`の出力`w2`は、MCTSシミュレーションによって得られた「MCTS winning probability」 [cite: 241, 25]
            # を指すので、そのMCTSの探索結果として、そのルートノードがどれだけ有望かという評価値を得る。
            # ここでは、MCTSが返す辞書の中から、最も勝率が高いと判断された手の勝率、または特定の評価値をMCTS勝率とする。
            # 簡潔化のため、MCTS.run_mctsは、特定の候補手に対するMCTSの勝率を直接返すものと仮定し、
            # MCTS内部でベストな子ノードの勝率を返すように修正する。
            
            # MCTS.run_mctsを呼び出す前に、`MCTS`クラスの`run_mcts`メソッドが
            # MCTSルートノード自体の勝率（そのノードの wins / visits）を返すように変更が必要。
            # または、`select_action`内でMCTSの根となるノードの子ノードの勝率を考慮する。
            
            # MCTSの`run_mcts`は、各候補手（ADPによって選択された手）に対応する盤面をMCTSのルートとして探索し、
            # そのMCTSの結果としての「勝率」を返すものとする。
            # MCTSの勝率は、そのMCTSの探索を完了した後の、ルートノード（ADPの候補手によって遷移した盤面）の評価値。
            # MCTSの評価は、そのルートノードの子ノードのQ値の最大値などで行われる。
            # 論文の`MTCS(m,s)`の出力`w2`は、MCTSシミュレーションによって得られた「MCTS winning probability」 [cite: 241, 25]
            # を指すので、そのMCTSの探索結果として、そのルートノードがどれだけ有望かという評価値を得る。
            # ここでは、MCTSが返す辞書の中から、最も勝率が高いと判断された手の勝率、または特定の評価値をMCTS勝率とする。
            # 簡潔化のため、MCTS.run_mctsは、特定の候補手に対するMCTSの勝率を直接返すものと仮定し、
            # MCTS内部でベストな子ノードの勝率を返すように修正する。
            
            mcts_win_prob = self.mcts.run_mcts_single_eval(temp_board_for_mcts, next_player_for_mcts)

            # 3. 最終予測勝率の計算: ADP勝率とMCTS勝率を重み付けして結合 [cite: 123, 359]
            # 論文の式(7): wp = λ * w1 + (1 - λ) * w2 [cite: 128, 364]
            # w1: ADPの勝率 [cite: 128, 364]
            # w2: MCTSの勝率 [cite: 128, 364]
            
            # MCTSの勝率 `mcts_win_prob` は、`next_player_for_mcts` が勝つ確率。
            # ADPの勝率 `adp_win_probabilities[i]` は、`current_player` が勝つ確率。
            # したがって、`mcts_win_prob` を `current_player` 視点に合わせるには `1 - mcts_win_prob` とすべき。
            # 論文の記述「final win rate both from ADP and MCTS algorithms」 [cite: 55, 291]
            # 「winRate against ADP depends on λ」 [cite: 144, 380] から、
            # どちらも「そのプレイヤーが勝つ確率」として扱っていると解釈するのが自然。
            # そのため、MCTSのシミュレーション報酬を「MCTSのルートノードのプレイヤーにとっての報酬」として定義する。
            # そうすることで、`mcts_win_prob` は ADPの勝率と同様に「現在のプレイヤーが勝つ確率」となる。

            combined_win_prob = self.lambda_param * adp_win_probabilities[i] + \
                                (1 - self.lambda_param) * mcts_win_prob
            
            final_predictions.append((move, combined_win_prob))
            
        # 4. 最も高い最終予測勝率を持つ手を選択 [cite: 26, 262]
        if not final_predictions:
            return None # 候補手がない場合

        best_move, _ = max(final_predictions, key=lambda x: x[1])
        return best_move

# MCTSクラスのrun_mctsを修正し、MCTSStageの挙動に合わせる
class MCTS:
    def __init__(self, board_size=15, simulation_times=400, heuristic_knowledge_enabled=True):
        self.board_size = board_size
        self.simulation_times = simulation_times
        self.heuristic_knowledge_enabled = heuristic_knowledge_enabled

    def _apply_heuristic_knowledge(self, board_state, current_player):
        # 論文のAlgorithm 1とAlgorithm 2で述べられているヒューリスティック知識を適用 [cite: 68, 304, 88, 324]
        # 具体的なルール: [cite: 71, 307]
        # 1. 自分の4連がある場合、5連になる位置に強制的に移動 [cite: 71, 307]
        # 2. 相手の4連がある場合、5連をブロックする位置に強制的に移動 [cite: 72, 308]
        # 3. 自分の3連がある場合、4連になる位置に強制的に移動 [cite: 73, 309]
        # 4. 相手の3連がある場合、4連をブロックする位置に強制的に移動 [cite: 76, 312]
        
        # このメソッドは、最も優先度の高い強制的な手があればそれを返し、なければNoneを返す。
        
        temp_board_obj = Board()
        temp_board_obj.MakeBoard(self.board_size, self.board_size)
        
        legal_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == 0:
                    legal_moves.append((r, c))
        random.shuffle(legal_moves)  # ランダムに合法手をシャッフル

        # 優先度1: 自分の5連を完成させる手
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(current_player)):
                return (r, c)
        
        # 優先度2: 相手の5連をブロックする手
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(opponent_player)):
                return (r, c)
        
        # 優先度3: 自分の3連から4連を作る手 [cite: 73, 309]
        # 各方向（水平、垂直、斜め）で3連のパターンを探す
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連から4連になるか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 4連が形成される場合、このセルを優先
                if count == 4:
                    # 端点が空いているか確認（オープン4の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        # 優先度4: 相手の3連をブロックする手 [cite: 76, 312]
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            # まず相手がこの位置に石を置いたと仮定
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連を形成するか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 相手が3連または4連を形成する場合、このセルをブロックする
                if count >= 3:
                    # 端点が空いているか確認（オープン3の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        return None # 強制手が見つからなかった場合

    def _simulate_game(self, board_state, starting_player):
        # シミュレーションステージ: リーフノードからゲームを終了までランダムにプレイアウトする [cite: 63, 299]
        # 論文のSimulation関数に相当 [cite: 68, 304]
        
        current_board = copy.deepcopy(board_state)
        current_player = starting_player
        
        temp_env = GomokuEnv(board_size=self.board_size)
        
        while True:
            temp_env.board.SetBoard(current_board)
            
            # 勝利判定 [cite: 68, 304]
            if temp_env.board.CheckWin(Stone.BLACK):
                return 1.0 if starting_player == 1 else 0.0 # 黒が勝ち、報酬を調整
            if temp_env.board.CheckWin(Stone.WHITE):
                return 1.0 if starting_player == 2 else 0.0 # 白が勝ち、報酬を調整
            
            legal_moves = []
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if current_board[r][c] == 0:
                        legal_moves.append((r, c))
            random.shuffle(legal_moves)  # ランダムに合法手をシャッフル

            if not legal_moves: # 引き分け [cite: 79, 315]
                return 0.0 # 引き分けは報酬0 [cite: 79, 315]

            chosen_move = None
            if self.heuristic_knowledge_enabled:
                # 強制手があればそれを選択 [cite: 68, 304]
                forced_action = self._apply_heuristic_knowledge(current_board, current_player)
                if forced_action and forced_action in legal_moves: # 強制手が合法手であることも確認
                    chosen_move = forced_action
            
            if chosen_move is None:
                chosen_move = random.choice(legal_moves) # 強制手がない場合、ランダムに選択 [cite: 68, 304]
            
            current_board[chosen_move[0]][chosen_move[1]] = current_player
            current_player = 3 - current_player # プレイヤー交代

    def _backpropagate(self, node, reward):
        # 逆伝播ステージ: シミュレーション結果を選択されたノードに逆伝播させ、状態値を更新する [cite: 64, 300]
        # 論文のBack Update関数に相当 [cite: 92, 328]
        
        while node is not None:
            node.visits += 1  # [cite: 92, 328]
            
            # 報酬の調整：親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換する必要がある。
            # 例えば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
            # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
            if node.current_player == (3 - (node.parent.current_player if node.parent else starting_player)):
                # ノードが親から遷移してきた手番のプレイヤー視点の場合
                # Simulatrionの報酬は、シミュレーションを開始したノードの`current_player`視点での勝敗を返す。
                # Backpropagationで親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換されるべき。
                # たとえば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
                # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
                node.wins += reward
            else:
                # 相手の視点の場合
                node.wins += (1.0 - reward)
            
            node = node.parent

    def run_mcts(self, root_board_state, root_player):
        # MCTSの主要な処理を統合
        # root_board_state: MCTSのルートノードとなる盤面状態
        # root_player: ルートノードでの手番のプレイヤー
        
        root_node = MCTSNode(root_board_state, current_player=root_player)

        for _ in range(self.simulation_times):
            # 1. 選択 (Selection) [cite: 60, 296, 61, 297]
            node = root_node
            # 全ての子ノードが展開されているか、終端ノードに到達するまで深く潜る
            while node.untried_moves == [] and node.children != {}:
                if node.is_terminal(): # 終端ノードに到達したらループを抜ける
                    break
                node = node.best_child()
            
            # 2. 展開 (Expansion) [cite: 62, 298]
            # 全ての子ノードが展開されていない場合、未試行の合法手から一つ選んで新しい子ノードを作成
            if node.untried_moves != [] and not node.is_terminal():
                node = node.expand()

            # 3. シミュレーション (Simulation) [cite: 63, 299]
            # リーフノードからゲームを終了までランダムにプレイアウトする
            # シミュレーションは展開されたノードから開始される
            # ここでのwinnerは「シミュレーションを開始したノードのプレイヤー」が勝った場合1、負けた場合0
            winner_reward = self._simulate_game(node.board_state, node.current_player)
            
            # 4. 逆伝播 (Backpropagation) [cite: 64, 300]
            # シミュレーション結果を選択されたノードに逆伝播させ、状態値を更新する
            self._backpropagate(node, winner_reward)

        # MCTSが終了したら、最も訪問回数が多い（または勝率が高い）ルートの子ノードに対応する手を返す
        # 論文のAlgorithm 2のBestChild(v0) [cite: 92, 328]
        if not root_node.children:
            return 0.0 # 候補手がない場合 (ゲームが既に終了している場合など)
            
        # 論文では、MCTSの勝率は wins / visits で表される [cite: 85, 321]。
        # ここでは、ルートノードの子ノード（各候補手）の勝率のうち、最も良いものを返す。
        # または、最終的に選択されるアクションのMCTS勝率を返す。
        
        # MCTSの勝率は、そのアクションを選択したルートノードの子ノードの`wins / visits`である。
        # 最終的にADPとMCTSを統合して手を選ぶため、ここでは各候補手に対するMCTS勝率を返すのではなく、
        # `ADPMCTS.select_action`で各候補手に対応するMCTS勝率を計算させる。
        # ここでは、指定されたルートからのMCTSの勝率として、ベストな子ノードの勝率を返す。
        
        # MCTSStageでは、各ADP候補手mに対して`MTCS(m,s)`から`w2`を得るとある [cite: 137, 373]。
        # これは、各ADP候補手を置いた後の盤面をMCTSのルートとして探索し、その探索結果としての勝率を意味する。
        # よって、ここではMCTSそのものの評価値を返す。
        
        # ここでは、ルートノードの各子ノード（つまり、可能な次の手）について、そのMCTSでの勝率を計算して返す
        mcts_win_probabilities = {}
        for move, child_node in root_node.children.items():
            if child_node.visits > 0:
                mcts_win_probabilities[move] = child_node.wins / child_node.visits
            else:
                mcts_win_probabilities[move] = 0.0 # 訪問されていないノードは勝率0

        return mcts_win_probabilities # 各候補手に対応するMCTS勝率の辞書


class ADPMCTS:
    def __init__(self, board_size=15, adp_model_path=None, mcts_simulation_times=400, lambda_param=0.5, num_adp_candidates=5):
        self.board_size = board_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adp_network = ADPNetwork(board_size=board_size, device=self.device)
        
        if adp_model_path:
            try:
                self.adp_network.load_state_dict(torch.load(adp_model_path, map_location=self.device))
                self.adp_network.eval() # 評価モードに設定
                print(f"ADPモデルを {adp_model_path} からロードしました。")
            except FileNotFoundError:
                print(f"警告: {adp_model_path} が見つかりませんでした。ADPモデルはランダムに初期化されます。")
            except Exception as e:
                print(f"警告: ADPモデルのロード中にエラーが発生しました: {e}。モデルはランダムに初期化されます。")
        else:
            print("警告: ADPモデルのパスが指定されていません。モデルはランダムに初期化されます。")

        self.mcts = MCTS(board_size=board_size, simulation_times=mcts_simulation_times)
        self.lambda_param = lambda_param # 論文のλパラメータ [cite: 128, 364]
        self.num_adp_candidates = num_adp_candidates # ADPから取得する候補手の数 [cite: 142, 378]
        
    def select_action(self, current_board_state, current_player):
        # 1. ADPステージ: ADPネットワークから上位5つの候補手とそのADP勝率を取得 [cite: 120, 356]
        # ADPStage(s0) [cite: 132, 368]
        adp_candidate_moves, adp_win_probabilities = self.adp_network.get_candidate_moves(
            current_board_state, current_player, self.num_adp_candidates
        )
        
        if not adp_candidate_moves:
            return None # 合法手がない場合

        final_predictions = []
        for i, move in enumerate(adp_candidate_moves):
            # 2. MCTSステージ: 各候補手に対してMCTSを実行し、MCTS勝率を取得 [cite: 122, 358]
            # MCTSStage(MADP) [cite: 132, 368]
            
            temp_board_for_mcts = copy.deepcopy(current_board_state)
            temp_board_for_mcts[move[0]][move[1]] = current_player
            
            # MCTSは、候補手を置いた後の盤面 (`temp_board_for_mcts`) をルートノードとし、
            # その盤面で手番を持つプレイヤー (`next_player_for_mcts`) の視点での勝率を計算する。
            next_player_for_mcts = 3 - current_player 
            
            # MCTS.run_mcts は、各合法手に対するMCTS勝率の辞書を返す。
            # ここでは、`move` (ADP候補手によって遷移した盤面) からのMCTS勝率を知りたい。
            # MCTSのルートノードが`temp_board_for_mcts`の場合、そのルートノードにおける勝率を評価する。
            
            # MCTS.run_mcts は、ルートノードの子ノードの勝率辞書を返すべきだが、
            # 論文の`MTCS(m,s)`は単一の勝率`w2`を返すことを示唆している [cite: 137, 373]。
            # ここでは、MCTSを一度実行して、そのルートノードの評価値（勝率）を得ると解釈する。
            # MCTSの評価は通常、ルートノードを十分に探索した後の、そのルートノード自体の評価である。
            # そのため、MCTSクラスには、ルートノードの勝率を計算するメソッドを追加する。
            # ただし、現状のMCTS.run_mctsは子ノードの勝率辞書を返すので、
            # その辞書から、そのMCTSのルートノード自体の勝率を評価するような適切な値を取得する必要がある。
            # 最も一般的なMCTSの評価は、ルートノードの最も有望な子ノードの勝率、またはルートノード自体のQ値。
            # 論文の`MTCS(m,s)`の出力`w2`は、MCTSシミュレーションによって得られた「MCTS winning probability」 [cite: 241, 25]
            # を指すので、そのMCTSの探索結果として、そのルートノードがどれだけ有望かという評価値を得る。
            # ここでは、MCTSが返す辞書の中から、最も勝率が高いと判断された手の勝率、または特定の評価値をMCTS勝率とする。
            # 簡潔化のため、MCTS.run_mctsは、特定の候補手に対するMCTSの勝率を直接返すものと仮定し、
            # MCTS内部でベストな子ノードの勝率を返すように修正する。
            
            # MCTS.run_mctsを呼び出す前に、`MCTS`クラスの`run_mcts`メソッドが
            # MCTSルートノード自体の勝率（そのノードの wins / visits）を返すように変更が必要。
            # または、`select_action`内でMCTSの根となるノードの子ノードの勝率を考慮する。
            
            # MCTSの`run_mcts`は、各候補手（ADPによって選択された手）に対応する盤面をMCTSのルートとして探索し、
            # そのMCTSの結果としての「勝率」を返すものとする。
            # MCTSの勝率は、そのMCTSの探索を完了した後の、ルートノード（ADPの候補手によって遷移した盤面）の評価値。
            # MCTSの評価は、そのルートノードの子ノードのQ値の最大値などで行われる。
            # 論文の`MTCS(m,s)`の出力`w2`は、MCTSシミュレーションによって得られた「MCTS winning probability」 [cite: 241, 25]
            # を指すので、そのMCTSの探索結果として、そのルートノードがどれだけ有望かという評価値を得る。
            # ここでは、MCTSが返す辞書の中から、最も勝率が高いと判断された手の勝率、または特定の評価値をMCTS勝率とする。
            # 簡潔化のため、MCTS.run_mctsは、特定の候補手に対するMCTSの勝率を直接返すものと仮定し、
            # MCTS内部でベストな子ノードの勝率を返すように修正する。
            
            mcts_win_prob = self.mcts.run_mcts_single_eval(temp_board_for_mcts, next_player_for_mcts)

            # 3. 最終予測勝率の計算: ADP勝率とMCTS勝率を重み付けして結合 [cite: 123, 359]
            # 論文の式(7): wp = λ * w1 + (1 - λ) * w2 [cite: 128, 364]
            # w1: ADPの勝率 [cite: 128, 364]
            # w2: MCTSの勝率 [cite: 128, 364]
            
            # MCTSの勝率 `mcts_win_prob` は、`next_player_for_mcts` が勝つ確率。
            # ADPの勝率 `adp_win_probabilities[i]` は、`current_player` が勝つ確率。
            # したがって、`mcts_win_prob` を `current_player` 視点に合わせるには `1 - mcts_win_prob` とすべき。
            # 論文の記述「final win rate both from ADP and MCTS algorithms」 [cite: 55, 291]
            # 「winRate against ADP depends on λ」 [cite: 144, 380] から、
            # どちらも「そのプレイヤーが勝つ確率」として扱っていると解釈するのが自然。
            # そのため、MCTSのシミュレーション報酬を「MCTSのルートノードのプレイヤーにとっての報酬」として定義する。
            # そうすることで、`mcts_win_prob` は ADPの勝率と同様に「現在のプレイヤーが勝つ確率」となる。

            combined_win_prob = self.lambda_param * adp_win_probabilities[i] + \
                                (1 - self.lambda_param) * mcts_win_prob
            
            final_predictions.append((move, combined_win_prob))
            
        # 4. 最も高い最終予測勝率を持つ手を選択 [cite: 26, 262]
        if not final_predictions:
            return None # 候補手がない場合

        best_move, _ = max(final_predictions, key=lambda x: x[1])
        return best_move

# MCTSクラスのrun_mctsを修正し、MCTSStageの挙動に合わせる
class MCTS:
    def __init__(self, board_size=15, simulation_times=400, heuristic_knowledge_enabled=True):
        self.board_size = board_size
        self.simulation_times = simulation_times
        self.heuristic_knowledge_enabled = heuristic_knowledge_enabled

    def _apply_heuristic_knowledge(self, board_state, current_player):
        # 論文のAlgorithm 1とAlgorithm 2で述べられているヒューリスティック知識を適用 [cite: 68, 304, 88, 324]
        # 具体的なルール: [cite: 71, 307]
        # 1. 自分の4連がある場合、5連になる位置に強制的に移動 [cite: 71, 307]
        # 2. 相手の4連がある場合、5連をブロックする位置に強制的に移動 [cite: 72, 308]
        # 3. 自分の3連がある場合、4連になる位置に強制的に移動 [cite: 73, 309]
        # 4. 相手の3連がある場合、4連をブロックする位置に強制的に移動 [cite: 76, 312]
        
        # このメソッドは、最も優先度の高い強制的な手があればそれを返し、なければNoneを返す。
        
        temp_board_obj = Board()
        temp_board_obj.MakeBoard(self.board_size, self.board_size)
        
        legal_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == 0:
                    legal_moves.append((r, c))
        random.shuffle(legal_moves)  # ランダムに合法手をシャッフル

        # 優先度1: 自分の5連を完成させる手
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(current_player)):
                return (r, c)
        
        # 優先度2: 相手の5連をブロックする手
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(opponent_player)):
                return (r, c)
        
        # 優先度3: 自分の3連から4連を作る手 [cite: 73, 309]
        # 各方向（水平、垂直、斜め）で3連のパターンを探す
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連から4連になるか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 4連が形成される場合、このセルを優先
                if count == 4:
                    # 端点が空いているか確認（オープン4の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        # 優先度4: 相手の3連をブロックする手 [cite: 76, 312]
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            # まず相手がこの位置に石を置いたと仮定
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連を形成するか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 相手が3連または4連を形成する場合、このセルをブロックする
                if count >= 3:
                    # 端点が空いているか確認（オープン3の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        return None # 強制手が見つからなかった場合

    def _simulate_game(self, board_state, starting_player):
        # シミュレーションステージ: リーフノードからゲームを終了までランダムにプレイアウトする [cite: 63, 299]
        # 論文のSimulation関数に相当 [cite: 68, 304]
        
        current_board = copy.deepcopy(board_state)
        current_player = starting_player
        
        temp_env = GomokuEnv(board_size=self.board_size)
        
        while True:
            temp_env.board.SetBoard(current_board)
            
            # 勝利判定 [cite: 68, 304]
            if temp_env.board.CheckWin(Stone.BLACK):
                return 1.0 if starting_player == 1 else 0.0 # 黒が勝ち、報酬を調整
            if temp_env.board.CheckWin(Stone.WHITE):
                return 1.0 if starting_player == 2 else 0.0 # 白が勝ち、報酬を調整
            
            legal_moves = []
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if current_board[r][c] == 0:
                        legal_moves.append((r, c))
            random.shuffle(legal_moves)  # ランダムに合法手をシャッフル

            if not legal_moves: # 引き分け [cite: 79, 315]
                return 0.0 # 引き分けは報酬0 [cite: 79, 315]

            chosen_move = None
            if self.heuristic_knowledge_enabled:
                # 強制手があればそれを選択 [cite: 68, 304]
                forced_action = self._apply_heuristic_knowledge(current_board, current_player)
                if forced_action and forced_action in legal_moves: # 強制手が合法手であることも確認
                    chosen_move = forced_action
            
            if chosen_move is None:
                chosen_move = random.choice(legal_moves) # 強制手がない場合、ランダムに選択 [cite: 68, 304]
            
            current_board[chosen_move[0]][chosen_move[1]] = current_player
            current_player = 3 - current_player # プレイヤー交代

    def _backpropagate(self, node, reward):
        # 逆伝播ステージ: シミュレーション結果を選択されたノードに逆伝播させ、状態値を更新する [cite: 64, 300]
        # 論文のBack Update関数に相当 [cite: 92, 328]
        
        while node is not None:
            node.visits += 1  # [cite: 92, 328]
            
            # 報酬の調整：親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換する必要がある。
            # 例えば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
            # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
            if node.current_player == (3 - (node.parent.current_player if node.parent else starting_player)):
                # ノードが親から遷移してきた手番のプレイヤー視点の場合
                # Simulatrionの報酬は、シミュレーションを開始したノードの`current_player`視点での勝敗を返す。
                # Backpropagationで親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換されるべき。
                # たとえば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
                # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
                node.wins += reward
            else:
                # 相手の視点の場合
                node.wins += (1.0 - reward)
            
            node = node.parent

    def run_mcts(self, root_board_state, root_player):
        # MCTSの主要な処理を統合
        # root_board_state: MCTSのルートノードとなる盤面状態
        # root_player: ルートノードでの手番のプレイヤー
        
        root_node = MCTSNode(root_board_state, current_player=root_player)

        for _ in range(self.simulation_times):
            # 1. 選択 (Selection) [cite: 60, 296, 61, 297]
            node = root_node
            # 全ての子ノードが展開されているか、終端ノードに到達するまで深く潜る
            while node.untried_moves == [] and node.children != {}:
                if node.is_terminal(): # 終端ノードに到達したらループを抜ける
                    break
                node = node.best_child()
            
            # 2. 展開 (Expansion) [cite: 62, 298]
            # 全ての子ノードが展開されていない場合、未試行の合法手から一つ選んで新しい子ノードを作成
            if node.untried_moves != [] and not node.is_terminal():
                node = node.expand()

            # 3. シミュレーション (Simulation) [cite: 63, 299]
            # リーフノードからゲームを終了までランダムにプレイアウトする
            # シミュレーションは展開されたノードから開始される
            # ここでのwinnerは「シミュレーションを開始したノードのプレイヤー」が勝った場合1、負けた場合0
            winner_reward = self._simulate_game(node.board_state, node.current_player)
            
            # 4. 逆伝播 (Backpropagation) [cite: 64, 300]
            # シミュレーション結果を選択されたノードに逆伝播させ、状態値を更新する
            self._backpropagate(node, winner_reward)

        # MCTSが終了したら、最も訪問回数が多い（または勝率が高い）ルートの子ノードに対応する手を返す
        # 論文のAlgorithm 2のBestChild(v0) [cite: 92, 328]
        if not root_node.children:
            return 0.0 # 候補手がない場合 (ゲームが既に終了している場合など)
            
        # 論文では、MCTSの勝率は wins / visits で表される [cite: 85, 321]。
        # ここでは、ルートノードの子ノード（各候補手）の勝率のうち、最も良いものを返す。
        # または、最終的に選択されるアクションのMCTS勝率を返す。
        
        # MCTSの勝率は、そのアクションを選択したルートノードの子ノードの`wins / visits`である。
        # 最終的にADPとMCTSを統合して手を選ぶため、ここでは各候補手に対するMCTS勝率を返すのではなく、
        # `ADPMCTS.select_action`で各候補手に対応するMCTS勝率を計算させる。
        # ここでは、指定されたルートからのMCTSの勝率として、ベストな子ノードの勝率を返す。
        
        # MCTSStageでは、各ADP候補手mに対して`MTCS(m,s)`から`w2`を得るとある [cite: 137, 373]。
        # これは、各ADP候補手を置いた後の盤面をMCTSのルートとして探索し、その探索結果としての勝率を意味する。
        # よって、ここではMCTSそのものの評価値を返す。
        
        # ここでは、ルートノードの各子ノード（つまり、可能な次の手）について、そのMCTSでの勝率を計算して返す
        mcts_win_probabilities = {}
        for move, child_node in root_node.children.items():
            if child_node.visits > 0:
                mcts_win_probabilities[move] = child_node.wins / child_node.visits
            else:
                mcts_win_probabilities[move] = 0.0 # 訪問されていないノードは勝率0

        return mcts_win_probabilities # 各候補手に対応するMCTS勝率の辞書


class ADPMCTS:
    def __init__(self, board_size=15, adp_model_path=None, mcts_simulation_times=400, lambda_param=0.5, num_adp_candidates=5):
        self.board_size = board_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adp_network = ADPNetwork(board_size=board_size, device=self.device)
        
        if adp_model_path:
            try:
                self.adp_network.load_state_dict(torch.load(adp_model_path, map_location=self.device))
                self.adp_network.eval() # 評価モードに設定
                print(f"ADPモデルを {adp_model_path} からロードしました。")
            except FileNotFoundError:
                print(f"警告: {adp_model_path} が見つかりませんでした。ADPモデルはランダムに初期化されます。")
            except Exception as e:
                print(f"警告: ADPモデルのロード中にエラーが発生しました: {e}。モデルはランダムに初期化されます。")
        else:
            print("警告: ADPモデルのパスが指定されていません。モデルはランダムに初期化されます。")

        self.mcts = MCTS(board_size=board_size, simulation_times=mcts_simulation_times)
        self.lambda_param = lambda_param # 論文のλパラメータ [cite: 128, 364]
        self.num_adp_candidates = num_adp_candidates # ADPから取得する候補手の数 [cite: 142, 378]
        
    def select_action(self, current_board_state, current_player):
        # 1. ADPステージ: ADPネットワークから上位5つの候補手とそのADP勝率を取得 [cite: 120, 356]
        # ADPStage(s0) [cite: 132, 368]
        adp_candidate_moves, adp_win_probabilities = self.adp_network.get_candidate_moves(
            current_board_state, current_player, self.num_adp_candidates
        )
        
        if not adp_candidate_moves:
            return None # 合法手がない場合

        final_predictions = []
        for i, move in enumerate(adp_candidate_moves):
            # 2. MCTSステージ: 各候補手に対してMCTSを実行し、MCTS勝率を取得 [cite: 122, 358]
            # MCTSStage(MADP) [cite: 132, 368]
            
            temp_board_for_mcts = copy.deepcopy(current_board_state)
            temp_board_for_mcts[move[0]][move[1]] = current_player
            
            # MCTSは、候補手を置いた後の盤面 (`temp_board_for_mcts`) をルートノードとし、
            # その盤面で手番を持つプレイヤー (`next_player_for_mcts`) の視点での勝率を計算する。
            next_player_for_mcts = 3 - current_player 
            
            # MCTS.run_mcts は、各合法手に対するMCTS勝率の辞書を返す。
            # ここでは、`move` (ADP候補手によって遷移した盤面) からのMCTS勝率を知りたい。
            # MCTSのルートノードが`temp_board_for_mcts`の場合、そのルートノードにおける勝率を評価する。
            
            # MCTS.run_mcts は、ルートノードの子ノードの勝率辞書を返すべきだが、
            # 論文の`MTCS(m,s)`は単一の勝率`w2`を返すことを示唆している [cite: 137, 373]。
            # ここでは、MCTSを一度実行して、そのルートノードの評価値（勝率）を得ると解釈する。
            # MCTSの評価は通常、ルートノードを十分に探索した後の、そのルートノード自体の評価である。
            # そのため、MCTSクラスには、ルートノードの勝率を計算するメソッドを追加する。
            # ただし、現状のMCTS.run_mctsは子ノードの勝率辞書を返すので、
            # その辞書から、そのMCTSのルートノード自体の勝率を評価するような適切な値を取得する必要がある。
            # 最も一般的なMCTSの評価は、ルートノードの最も有望な子ノードの勝率、またはルートノード自体のQ値。
            # 論文の`MTCS(m,s)`の出力`w2`は、MCTSシミュレーションによって得られた「MCTS winning probability」 [cite: 241, 25]
            # を指すので、そのMCTSの探索結果として、そのルートノードがどれだけ有望かという評価値を得る。
            # ここでは、MCTSが返す辞書の中から、最も勝率が高いと判断された手の勝率、または特定の評価値をMCTS勝率とする。
            # 簡潔化のため、MCTS.run_mctsは、特定の候補手に対するMCTSの勝率を直接返すものと仮定し、
            # MCTS内部でベストな子ノードの勝率を返すように修正する。
            
            # MCTS.run_mctsを呼び出す前に、`MCTS`クラスの`run_mcts`メソッドが
            # MCTSルートノード自体の勝率（そのノードの wins / visits）を返すように変更が必要。
            # または、`select_action`内でMCTSの根となるノードの子ノードの勝率を考慮する。
            
            # MCTSの`run_mcts`は、各候補手（ADPによって選択された手）に対応する盤面をMCTSのルートとして探索し、
            # そのMCTSの結果としての「勝率」を返すものとする。
            # MCTSの勝率は、そのMCTSの探索を完了した後の、ルートノード（ADPの候補手によって遷移した盤面）の評価値。
            # MCTSの評価は、そのルートノードの子ノードのQ値の最大値などで行われる。
            # 論文の`MTCS(m,s)`の出力`w2`は、MCTSシミュレーションによって得られた「MCTS winning probability」 [cite: 241, 25]
            # を指すので、そのMCTSの探索結果として、そのルートノードがどれだけ有望かという評価値を得る。
            # ここでは、MCTSが返す辞書の中から、最も勝率が高いと判断された手の勝率、または特定の評価値をMCTS勝率とする。
            # 簡潔化のため、MCTS.run_mctsは、特定の候補手に対するMCTSの勝率を直接返すものと仮定し、
            # MCTS内部でベストな子ノードの勝率を返すように修正する。
            
            mcts_win_prob = self.mcts.run_mcts_single_eval(temp_board_for_mcts, next_player_for_mcts)

            # 3. 最終予測勝率の計算: ADP勝率とMCTS勝率を重み付けして結合 [cite: 123, 359]
            # 論文の式(7): wp = λ * w1 + (1 - λ) * w2 [cite: 128, 364]
            # w1: ADPの勝率 [cite: 128, 364]
            # w2: MCTSの勝率 [cite: 128, 364]
            
            # MCTSの勝率 `mcts_win_prob` は、`next_player_for_mcts` が勝つ確率。
            # ADPの勝率 `adp_win_probabilities[i]` は、`current_player` が勝つ確率。
            # したがって、`mcts_win_prob` を `current_player` 視点に合わせるには `1 - mcts_win_prob` とすべき。
            # 論文の記述「final win rate both from ADP and MCTS algorithms」 [cite: 55, 291]
            # 「winRate against ADP depends on λ」 [cite: 144, 380] から、
            # どちらも「そのプレイヤーが勝つ確率」として扱っていると解釈するのが自然。
            # そのため、MCTSのシミュレーション報酬を「MCTSのルートノードのプレイヤーにとっての報酬」として定義する。
            # そうすることで、`mcts_win_prob` は ADPの勝率と同様に「現在のプレイヤーが勝つ確率」となる。

            combined_win_prob = self.lambda_param * adp_win_probabilities[i] + \
                                (1 - self.lambda_param) * mcts_win_prob
            
            final_predictions.append((move, combined_win_prob))
            
        # 4. 最も高い最終予測勝率を持つ手を選択 [cite: 26, 262]
        if not final_predictions:
            return None # 候補手がない場合

        best_move, _ = max(final_predictions, key=lambda x: x[1])
        return best_move

# MCTSクラスのrun_mctsを修正し、MCTSStageの挙動に合わせる
class MCTS:
    def __init__(self, board_size=15, simulation_times=400, heuristic_knowledge_enabled=True):
        self.board_size = board_size
        self.simulation_times = simulation_times
        self.heuristic_knowledge_enabled = heuristic_knowledge_enabled

    def _apply_heuristic_knowledge(self, board_state, current_player):
        # 論文のAlgorithm 1とAlgorithm 2で述べられているヒューリスティック知識を適用 [cite: 68, 304, 88, 324]
        # 具体的なルール: [cite: 71, 307]
        # 1. 自分の4連がある場合、5連になる位置に強制的に移動 [cite: 71, 307]
        # 2. 相手の4連がある場合、5連をブロックする位置に強制的に移動 [cite: 72, 308]
        # 3. 自分の3連がある場合、4連になる位置に強制的に移動 [cite: 73, 309]
        # 4. 相手の3連がある場合、4連をブロックする位置に強制的に移動 [cite: 76, 312]
        
        # このメソッドは、最も優先度の高い強制的な手があればそれを返し、なければNoneを返す。
        
        temp_board_obj = Board()
        temp_board_obj.MakeBoard(self.board_size, self.board_size)
        
        legal_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == 0:
                    legal_moves.append((r, c))
        random.shuffle(legal_moves)  # ランダムに合法手をシャッフル
        # 優先度1: 自分の5連を完成させる手
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(current_player)):
                return (r, c)
        
        # 優先度2: 相手の5連をブロックする手
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            temp_board_obj.SetBoard(temp_board)
            if temp_board_obj.CheckWin(Stone(opponent_player)):
                return (r, c)
        
        # 優先度3: 自分の3連から4連を作る手 [cite: 73, 309]
        # 各方向（水平、垂直、斜め）で3連のパターンを探す
        for r, c in legal_moves:
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = current_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連から4連になるか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == current_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 4連が形成される場合、このセルを優先
                if count == 4:
                    # 端点が空いているか確認（オープン4の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        # 優先度4: 相手の3連をブロックする手 [cite: 76, 312]
        opponent_player = 3 - current_player
        for r, c in legal_moves:
            # まず相手がこの位置に石を置いたと仮定
            temp_board = copy.deepcopy(board_state)
            temp_board[r][c] = opponent_player
            
            # 水平、垂直、対角線（右下がり、右上がり）の4方向について3連を形成するか確認
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 仮に置いた石から左右（または上下、斜め）に探索
                count = 1  # 置いた石自体をカウント
                # 順方向を探索
                nr, nc = r + dr, c + dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                
                # 逆方向を探索
                nr, nc = r - dr, c - dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr][nc] == opponent_player:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                
                # 相手が3連または4連を形成する場合、このセルをブロックする
                if count >= 3:
                    # 端点が空いているか確認（オープン3の場合）
                    nr1, nc1 = r + dr * count, c + dc * count
                    nr2, nc2 = r - dr * count, c - dc * count
                    if ((0 <= nr1 < self.board_size and 0 <= nc1 < self.board_size and temp_board[nr1][nc1] == 0) or
                        (0 <= nr2 < self.board_size and 0 <= nc2 < self.board_size and temp_board[nr2][nc2] == 0)):
                        return (r, c)
        
        return None # 強制手が見つからなかった場合

    def _simulate_game(self, board_state, starting_player):
        # シミュレーションステージ: リーフノードからゲームを終了までランダムにプレイアウトする [cite: 63, 299]
        # 論文のSimulation関数に相当 [cite: 68, 304]
        
        current_board = copy.deepcopy(board_state)
        current_player = starting_player
        
        temp_env = GomokuEnv(board_size=self.board_size)
        
        while True:
            temp_env.board.SetBoard(current_board)
            
            # 勝利判定 [cite: 68, 304]
            if temp_env.board.CheckWin(Stone.BLACK):
                return 1.0 if starting_player == 1 else 0.0 # 黒が勝ち、報酬を調整
            if temp_env.board.CheckWin(Stone.WHITE):
                return 1.0 if starting_player == 2 else 0.0 # 白が勝ち、報酬を調整
            
            legal_moves = []
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if current_board[r][c] == 0:
                        legal_moves.append((r, c))
            random.shuffle(legal_moves)  # ランダムに合法手をシャッフル

            if not legal_moves: # 引き分け [cite: 79, 315]
                return 0.0 # 引き分けは報酬0 [cite: 79, 315]

            chosen_move = None
            if self.heuristic_knowledge_enabled:
                # 強制手があればそれを選択 [cite: 68, 304]
                forced_action = self._apply_heuristic_knowledge(current_board, current_player)
                if forced_action and forced_action in legal_moves: # 強制手が合法手であることも確認
                    chosen_move = forced_action
            
            if chosen_move is None:
                chosen_move = random.choice(legal_moves) # 強制手がない場合、ランダムに選択 [cite: 68, 304]
            
            current_board[chosen_move[0]][chosen_move[1]] = current_player
            current_player = 3 - current_player # プレイヤー交代

    def _backpropagate(self, node, reward):
        # 逆伝播ステージ: シミュレーション結果を選択されたノードに逆伝播させ、状態値を更新する [cite: 64, 300]
        # 論文のBack Update関数に相当 [cite: 92, 328]
        
        while node is not None:
            node.visits += 1  # [cite: 92, 328]
            
            # 報酬の調整：親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換する必要がある。
            # 例えば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
            # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
            if node.current_player == (3 - (node.parent.current_player if node.parent else starting_player)):
                # ノードが親から遷移してきた手番のプレイヤー視点の場合
                # Simulatrionの報酬は、シミュレーションを開始したノードの`current_player`視点での勝敗を返す。
                # Backpropagationで親ノードに逆伝播する際、報酬は親ノードのプレイヤー視点に変換されるべき。
                # たとえば、子ノードが勝った場合、親ノード（相手のプレイヤー）から見れば負けである。
                # したがって、`wins`は常にそのノードの`current_player`にとっての勝数である。
                node.wins += reward
            else:
                # 相手の視点の場合
                node.wins += (1.0 - reward)
            
            node = node.parent

    def run_mcts_single_eval(self, root_board_state, root_player):
        # MCTSを単一のルートノードの評価のために実行し、そのルートノードの勝率を返す
        root_node = MCTSNode(root_board_state, current_player=root_player)

        for _ in range(self.simulation_times):
            node = root_node
            
            # Selection
            while node.untried_moves == [] and node.children != {}:
                if node.is_terminal():
                    break
                node = node.best_child()
            
            # Expansion
            if node.untried_moves != [] and not node.is_terminal():
                node = node.expand()

            # Simulation
            # simulationは展開されたノードから開始される
            winner_reward = self._simulate_game(node.board_state, node.current_player)
            
            # Backpropagation
            self._backpropagate(node, winner_reward)

        # MCTSが終了したら、ルートノードの勝率を返す
        # 論文の`w2`は、そのMCTSの根ノードからの勝率を意味する。
        return root_node.wins / root_node.visits if root_node.visits > 0 else 0.0


class ADPTrainer:
    def __init__(self, env: GomokuEnv, adp_network: ADPNetwork, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.env = env
        self.adp_network = adp_network
        self.optimizer = optim.Adam(adp_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss() # 価値関数の推定なのでMSELossを使用
        self.gamma = gamma # 割引率
        
        # ε-greedy用のパラメータ
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def train_one_episode(self):
        state = self.env.reset()
        done = False
        episode_history = [] # (state, action, next_state, reward, current_player) の履歴

        while not done:
            current_player = self.env.current_player
            board_state_np = state.cpu().numpy() # NumPyに変換してADPNetworkに渡す

            # ADPネットワークから候補手と勝率を取得 (行動選択ポリシー)
            # 論文では、Action Selectionがx(t)からu(t)を生成するとある [cite: 103, 339]。
            # ここでは、ADPNetworkのget_candidate_movesが、勝率の高い手を探索に利用する。
            #訓練時には、探索と活用のバランスを取るためにε-greedy戦略を使用。
            
            legal_moves = self.env.get_legal_moves()
            if not legal_moves:
                _, reward, done, info = self.env.step(None) # 引き分けとして終了
                break

            # ε-greedy法による行動選択
            if random.random() < self.epsilon:  # εの確率でランダムに行動
                action = random.choice(legal_moves)
            else:  # (1-ε)の確率で最も高い勝率の行動を選択
                action_values = []
                for r, c in legal_moves:
                    temp_board = copy.deepcopy(board_state_np)
                    temp_board[r][c] = current_player
                    prob = self.adp_network.predict_win_probability(temp_board, current_player)
                    action_values.append((prob, (r, c)))
                random.shuffle(action_values)  # ランダムにシャッフルして探索の多様性を持たせる
                action_values.sort(key=lambda x: x[0], reverse=True)
                action = action_values[0][1]  # 最も良い手を選択

            next_state_tensor, reward_tensor, done, info = self.env.step(action)
            
            # エピソード履歴に保存
            episode_history.append({
                'state': board_state_np,
                'action': action,
                'next_state': next_state_tensor.cpu().numpy(),
                'reward': reward_tensor.item(),
                'current_player': current_player
            })
            state = next_state_tensor
            
        # εを減少させる（学習が進むにつれて探索より活用を重視）
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
        # 逆伝播と学習 (Bellman方程式に基づく) [cite: 105, 341]
        self.optimizer.zero_grad()
        loss = 0
        
        # 最終状態の価値 (V(t+1)) は、ゲームの結果から直接得る
        # 訓練時のV(t+1)は、ゲームの最終結果に基づく報酬r(x(t+1)) [cite: 104, 340]
        # ただし、TD学習ではV(t+1)もCritic Networkの予測値を使用する。
        # V(t)とV(t+1)を用いてCritic Networkの重みを更新し、Bellman方程式を満たすようにする [cite: 105, 341]。
        
        # 論文のADP構造では、Critic Networkは V(t) と V(t+1) を推定し、報酬 r(x(t+1)) と共に
        # Bellman 方程式に基づいて重みを更新する [cite: 105, 341]。
        # 価値関数のターゲットは `r(x(t+1)) + gamma * V(t+1)`
        
        # エピソードを逆順にたどり、各ステップで価値関数を更新
        final_reward = episode_history[-1]['reward'] if episode_history else 0.0

        for i in reversed(range(len(episode_history))):
            current_data = episode_history[i]
            s_t_np = current_data['state']
            a_t = current_data['action']
            r_t_plus_1 = current_data['reward'] # このrewardは、`s_t_np`から`a_t`を打った後の報酬
            s_t_plus_1_np = current_data['next_state']
            player_t = current_data['current_player']

            # 現在の状態 `s_t_np` における価値 V(t) を Critic Network で予測
            v_t_pred = self.adp_network(self.adp_network.get_board_features(s_t_np, player_t).unsqueeze(0))

            # 次の状態 `s_t_plus_1_np` における価値 V(t+1) を Critic Network で予測
            # ゲームが終了している場合、V(t+1)は0 (終端状態)
            if done and i == len(episode_history) - 1: # 最後のステップ
                # 報酬を0-1の範囲に正規化
                normalized_reward = torch.clamp((torch.tensor(r_t_plus_1, dtype=torch.float32, device=self.adp_network.device) + 1.0) / 2.0, 0.0, 1.0)
                target_v_t = normalized_reward
            else:
                next_player_for_v_t_plus_1 = 3 - player_t # 次の状態での手番のプレイヤー
                
                # 価値関数は通常、現在のプレイヤーの視点での価値を予測する。
                # 次の盤面`s_t_plus_1_np`で手番が相手に移った場合、その相手の勝率が予測される。
                # しかし、我々のADPNetworkは引数`current_player`を受け取るので、
                # `s_t_plus_1_np`における`player_t`の価値を予測させるために、
                # その状態での`player_t`の石の種類と相手の石の種類を渡す必要がある。
                # より正確には、ADPNetworkは盤面の状態を受け取り、その状態における特定のプレイヤーの勝率を出すべき。
                
                # ここでは、V(t+1)を予測する際にも、`player_t`の視点での価値を予測させる。
                # つまり、`s_t_plus_1_np`盤面において、`player_t`がどれだけ有利か。
                # ただし、`next_state_tensor`は`3 - player_t`の手番になっているため、
                # ここで予測するV(t+1)は、`next_player_for_v_t_plus_1`の視点での価値となる。
                # 報酬が現在のプレイヤーの視点なので、価値も現在のプレイヤーの視点に揃える必要がある。
                
                # 論文のBellman方程式 V(t) = r(t+1) + gamma * V(t+1) は、
                # V(t)は現在のプレイヤーの価値、V(t+1)は次のプレイヤーの価値、と解釈されることが多い。
                # 二人零和ゲームでは、次のプレイヤーの価値は現在のプレイヤーの価値の逆なので、
                # V(t) = r(t+1) + gamma * (1 - V_next_player(t+1)) となる。
                
                v_t_plus_1_pred = self.adp_network(self.adp_network.get_board_features(s_t_plus_1_np, next_player_for_v_t_plus_1).unsqueeze(0))
                
                # 報酬を0-1の範囲に正規化し、ターゲット値もクリッピング
                normalized_reward = torch.clamp((torch.tensor(r_t_plus_1, dtype=torch.float32, device=self.adp_network.device) + 1.0) / 2.0, 0.0, 1.0)
                target_v_t = torch.clamp(normalized_reward + self.gamma * (1.0 - v_t_plus_1_pred), 0.0, 1.0)
            
            # TD誤差を計算
            td_error = target_v_t - v_t_pred
            # テンソルのサイズを一致させる
            target_v_t = target_v_t.reshape_as(v_t_pred)
            batch_loss = self.criterion(v_t_pred, target_v_t)
            loss += batch_loss
            
            # デバッグ情報（必要に応じてコメントアウト）
            # if i == 0 and (episode_history[0].get('debug_counter', 0) % 100 == 0):
            #     print(f"Debug - v_t_pred: {v_t_pred.item():.4f}, target_v_t: {target_v_t.item():.4f}, loss: {batch_loss.item():.4f}")

        if len(episode_history) > 0:
            loss.backward()
            # 勾配クリッピングを追加して勾配爆発を防止
            torch.nn.utils.clip_grad_norm_(self.adp_network.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return loss.item() if len(episode_history) > 0 else 0.0


class ParallelADPTrainer:
    def __init__(self, board_size=15, learning_rate=0.001, gamma=0.99, num_workers=4, 
                 batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.board_size = board_size
        self.device = device
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.gamma = gamma
        
        # メインプロセスでADPネットワークとオプティマイザを設定
        self.adp_network = ADPNetwork(board_size=board_size, device=device)
        self.optimizer = optim.Adam(self.adp_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # εグリーディの設定値
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.9995
        self.epsilon = self.epsilon_start
        
        # 損失の履歴を保存するリストを追加
        self.all_losses = []

    def worker_process(self, worker_id, episode_queue, result_queue, model_dict, epsilon):
        """ワーカープロセスがエピソードをシミュレーションし結果を返す関数"""
        # ワーカープロセス用のGomokuEnvとADPNetworkを初期化
        import torch.multiprocessing as mp
        
        # CPUデバイスでワーカープロセス用のネットワークを作成
        worker_device = "cpu"  # ワーカープロセスでは常にCPUを使用
        worker_network = ADPNetwork(board_size=self.board_size, device=worker_device)
        worker_network.load_state_dict({k: v.cpu() for k, v in model_dict.items()})
        worker_network.eval()  # 評価モードに設定（ここではモデル予測のみを使用）
        
        env = GomokuEnv(board_size=self.board_size, train_target="first", device=worker_device)
        
        while True:
            # キューからエピソード番号を取得するかNoneを受け取って終了
            episode_num = episode_queue.get()
            if episode_num is None:  # 終了シグナル
                break
                
            # エピソードを1つシミュレーション
            state = env.reset()
            done = False
            episode_data = []
            
            while not done:
                current_player = env.current_player
                board_state_np = state.cpu().numpy()
                
                # ε-greedy法による行動選択
                legal_moves = env.get_legal_moves()
                if not legal_moves:
                    _, reward, done, info = env.step(None)  # 引き分け
                    break
                
                if random.random() < epsilon:  # εの確率でランダムに行動
                    action = random.choice(legal_moves)
                else:  # (1-ε)の確率で最も高い勝率の行動を選択
                    action_values = []
                    for r, c in legal_moves:
                        temp_board = copy.deepcopy(board_state_np)
                        temp_board[r][c] = current_player
                        with torch.no_grad():
                            prob = worker_network.predict_win_probability(temp_board, current_player)
                        action_values.append((prob, (r, c)))
                    random.shuffle(action_values)  # ランダムにシャッフルして探索の多様性を持たせる
                    
                    action_values.sort(key=lambda x: x[0], reverse=True)
                    action = action_values[0][1]  # 最も良い手を選択
                
                next_state, reward, done, info = env.step(action)
                
                # エピソードデータを保存
                episode_data.append({
                    'state': board_state_np,
                    'action': action,
                    'next_state': next_state.cpu().numpy(),
                    'reward': reward.item(),
                    'current_player': current_player
                })
                
                state = next_state
            
            # エピソード結果をキューに送信
            result_queue.put((episode_num, episode_data))
            
    def train_parallel(self, num_episodes, checkpoint_interval=1000, checkpoint_dir='checkpoints'):
        """複数のプロセスを使って並列にトレーニングを実行"""
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)  # Windows/CUDAで必要
        
        # プロセス間で共有するキューを作成
        episode_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # エピソード番号をキューに追加
        for i in range(num_episodes):
            episode_queue.put(i)
            
        # 終了シグナルをワーカー数だけ追加
        for _ in range(self.num_workers):
            episode_queue.put(None)
            
        # モデルの状態辞書をコピー
        model_dict = {k: v.cpu() for k, v in self.adp_network.state_dict().items()}
        
        # ワーカープロセスを開始
        workers = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=self.worker_process,
                args=(i, episode_queue, result_queue, model_dict, self.epsilon)
            )
            p.start()
            workers.append(p)
            
        # 結果を処理してモデルを更新
        completed_episodes = 0
        losses = []
        
        while completed_episodes < num_episodes:
            # キューから結果を取得
            episode_num, episode_data = result_queue.get()
            completed_episodes += 1
            
            # モデルの更新（バックプロパゲーション）
            if episode_data:  # 有効なエピソードデータがある場合
                self.optimizer.zero_grad()
                episode_loss = self.update_model_from_episode(episode_data)
                losses.append(episode_loss)
                self.all_losses.append(episode_loss)  # 全体の損失履歴に追加
                
                # εを減少させる
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                
                # ワーカーの新しいεを更新（定期的に）
                if completed_episodes % 100 == 0:
                    # モデルの状態辞書を更新
                    model_dict = {k: v.cpu() for k, v in self.adp_network.state_dict().items()}
                    
                    # すべてのワーカーを終了して再起動（モデルとεを更新）
                    for p in workers:
                        p.terminate()
                        p.join()
                    
                    # 新しいキューを作成
                    episode_queue = mp.Queue()
                    for i in range(num_episodes - completed_episodes):
                        episode_queue.put(completed_episodes + i)
                    for _ in range(self.num_workers):
                        episode_queue.put(None)
                    
                    # 新しいワーカーを起動
                    workers = []
                    for i in range(self.num_workers):
                        p = mp.Process(
                            target=self.worker_process,
                            args=(i, episode_queue, result_queue, model_dict, self.epsilon)
                        )
                        p.start()
                        workers.append(p)
            
            # 進捗報告
            if (completed_episodes) % 100 == 0:
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"エピソード {completed_episodes}/{num_episodes}, 平均損失: {avg_loss:.4f}, ε: {self.epsilon:.4f}")
                losses = []  # リセット
                
            # チェックポイントの保存
            if completed_episodes % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"adp_model_episode_{completed_episodes}.pth")
                torch.save(self.adp_network.state_dict(), checkpoint_path)
                print(f"モデルを保存しました: {checkpoint_path}")
                
                # 損失のグラフを描画して保存
                self.plot_losses(completed_episodes, checkpoint_dir)
                
        # すべてのワーカープロセスを終了
        for p in workers:
            p.terminate()
            p.join()
            
        return self.adp_network
    
    def plot_losses(self, episode, checkpoint_dir):
        """損失の履歴をグラフとして描画し保存する"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.all_losses)), self.all_losses, alpha=0.6)
        
        # 移動平均を計算して描画
        window_size = min(100, len(self.all_losses))
        if window_size > 0:
            moving_avg = []
            for i in range(len(self.all_losses) - window_size + 1):
                moving_avg.append(sum(self.all_losses[i:i+window_size]) / window_size)
            plt.plot(range(window_size-1, len(self.all_losses)), moving_avg, 'r-', linewidth=2)
        
        plt.title(f'損失の推移 (エピソード {episode}まで)')
        plt.xlabel('エピソード')
        plt.ylabel('損失')
        plt.grid(True)
        
        # グラフを保存
        graph_path = os.path.join(checkpoint_dir, f'loss_graph_episode_{episode}.png')
        plt.savefig(graph_path)
        plt.close()
        print(f"損失グラフを保存しました: {graph_path}")

# ...existing code...

if __name__ == "__main__":
    BOARD_SIZE = 15
    ADP_MODEL_PATH = 'adp_model.pth'
    NUM_ADP_TRAINING_EPISODES = 2500000 # ADP訓練エピソード数
    MCTS_SIMULATION_TIMES = 1000  # MCTSシミュレーション回数 [cite: 163, 399]
    LAMBDA_PARAM = 0.5 # 論文の最適値 [cite: 138, 374]
    CHECKPOINT_INTERVAL = 1000  # この間隔でモデルを保存
    CHECKPOINT_DIR = 'checkpoints'  # チェックポイント保存ディレクトリ
    # 並列処理用のパラメータ
    NUM_WORKERS = max(1, os.cpu_count() - 1)  # CPU数-1（少なくとも1）をワーカー数に
    
    import os
    # チェックポイントディレクトリが存在しない場合は作成
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    print(f"並列処理ワーカー数: {NUM_WORKERS}")

    # Step 1: ADPNetworkの訓練 - 並列処理バージョン
    print("--- ADPネットワークの並列訓練開始 ---")
    
    # 並列トレーナーの初期化
    parallel_trainer = ParallelADPTrainer(
        board_size=BOARD_SIZE,
        learning_rate=0.0001,
        gamma=0.99,
        num_workers=NUM_WORKERS,
        device=device
    )
    
    # 学習再開のためのチェックポイントロード機能
    start_episode = 0
    latest_checkpoint = None
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('adp_model_episode_') and f.endswith('.pth')]
    # まずADPNetworkインスタンスを作成
    adp_network_for_training = ADPNetwork(board_size=BOARD_SIZE, device=device)
    if checkpoint_files:
        # 最新のチェックポイントを見つける
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[3].split('.')[0]))
        start_episode = int(latest_checkpoint.split('_')[3].split('.')[0])
        try:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
            adp_network_for_training.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"チェックポイント {checkpoint_path} をロードしました。エピソード {start_episode} から再開します。")
        except Exception as e:
            print(f"チェックポイントのロードに失敗しました: {e}")
            start_episode = 0

    adp_network_for_training.train() # 訓練モードに設定
    
    # トレーナーを1回だけ初期化して全エピソードで使用する
    trainer = ADPTrainer(env=GomokuEnv(board_size=BOARD_SIZE, train_target="first", device=device), 
                       adp_network=adp_network_for_training, 
                       learning_rate=0.0001,
                       epsilon_start=0.9,  # 初期εの値
                       epsilon_end=0.05,   # 最小εの値
                       epsilon_decay=0.9995)  # εの減衰率
                       
    # 損失の履歴を保存するリスト（非並列版用）
    all_losses = []

    for episode in range(start_episode, NUM_ADP_TRAINING_EPISODES):
        loss = trainer.train_one_episode()
        all_losses.append(loss)  # 損失を記録
        
        if (episode + 1) % 100 == 0:
            print(f"エピソード {episode + 1}/{NUM_ADP_TRAINING_EPISODES}, 損失: {loss:.4f}, ε: {trainer.epsilon:.4f}")
        
        # 定期的にモデルを保存
        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"adp_model_episode_{episode + 1}.pth")
            torch.save(adp_network_for_training.state_dict(), checkpoint_path)
            print(f"モデルを保存しました: {checkpoint_path}")
            
            # 損失のグラフを描画して保存（非並列版）
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(all_losses)), all_losses, alpha=0.6)
            
            # 移動平均の描画
            window_size = min(100, len(all_losses))
            if window_size > 0:
                moving_avg = []
                for i in range(len(all_losses) - window_size + 1):
                    moving_avg.append(sum(all_losses[i:i+window_size]) / window_size)
                plt.plot(range(window_size-1, len(all_losses)), moving_avg, 'r-', linewidth=2)
            
            plt.title(f'損失の推移 (全エピソード)')
            plt.xlabel('エピソード')
            plt.ylabel('損失')
            plt.grid(True)
            
            # 最終グラフを保存
            final_graph_path = os.path.join(CHECKPOINT_DIR, 'loss_graph_final.png')
            plt.savefig(final_graph_path)
            plt.close()
            print(f"最終損失グラフを保存しました: {final_graph_path}")
    
    # 訓練済みモデルの保存
    torch.save(adp_network_for_training.state_dict(), ADP_MODEL_PATH)
    print(f"訓練済みADPモデルを {ADP_MODEL_PATH} に保存しました。")
    
    # 最終的な損失グラフを保存
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(all_losses)), all_losses, alpha=0.6)
    
    # 移動平均の描画
    window_size = min(100, len(all_losses))
    if window_size > 0:
        moving_avg = []
        for i in range(len(all_losses) - window_size + 1):
            moving_avg.append(sum(all_losses[i:i+window_size]) / window_size)
        plt.plot(range(window_size-1, len(all_losses)), moving_avg, 'r-', linewidth=2)
    
    plt.title(f'損失の推移 (全エピソード)')
    plt.xlabel('エピソード')
    plt.ylabel('損失')
    plt.grid(True)
    
    # 最終グラフを保存
    final_graph_path = os.path.join(CHECKPOINT_DIR, 'loss_graph_final.png')
    plt.savefig(final_graph_path)
    plt.close()
    print(f"最終損失グラフを保存しました: {final_graph_path}")
    
    adp_network_for_training.eval()  # 評価モードに設定
    
    # Step 2: ADPMCTSプレイヤーの初期化（ここで追加）
    adpmcts_player = ADPMCTS(
        board_size=BOARD_SIZE,
        adp_model_path=ADP_MODEL_PATH,
        mcts_simulation_times=MCTS_SIMULATION_TIMES,
        lambda_param=LAMBDA_PARAM
    )

    # Step 3: AIと人間の対局
    print("\n--- 五目並べゲーム開始 (AI vs 人間) ---")
    game_env = GomokuEnv(board_size=BOARD_SIZE, train_target="first", device=device) # AIが先手 (黒)
    current_board_state_tensor = game_env.reset()
    current_board_state_np = current_board_state_tensor.cpu().numpy()
    done = False
    turn_count = 0

    print("AI (X) vs 人間 (O)")
    game_env.render()

    while not done:
        player_name = "AI (X)" if game_env.current_player == 1 else "人間 (O)"
        print(f"\n--- {player_name} の番です ---")
        
        action = None
        if game_env.current_player == 1: # AI (黒) のターン
            print("AI思考中...")

            # ADPMCTSプレイヤーを使用して手を選択
            action = adpmcts_player.select_action(current_board_state_np, game_env.current_player)
            
            if action is None:
                print("AIに合法手が見つかりませんでした。ゲーム終了。")
                done = True
                break
            
            print(f"AIが選択した手: {action}")
        else: # 人間 (白) のターン
            action = game_env.get_human_action()
            
        current_board_state_tensor, reward, done, info = game_env.step(action)
        current_board_state_np = current_board_state_tensor.cpu().numpy()
        game_env.render()
        turn_count += 1

        if done:
            if info.get("invalid_action"):
                print(f"無効な手が打たれました: {info.get('reason', '不明')}")