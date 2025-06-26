import numpy as np
import torch
import collections

import time

from N_Tuple.ReplayBuffer import Experience

INF = 100

# N-tupleの取り出し座標を作成
def make_ntuple(n: int, x: int, y: int, dx: int, dy: int, device: torch.device) -> torch.Tensor:
    ntuple = torch.zeros((n, 2), dtype=torch.long, device=device)
    for i in range(n):
        ntuple[i] = torch.tensor([x + i * dx, y + i * dy], dtype=torch.long, device=device)
        
    return ntuple

# 盤面に存在しうる縦横斜め方向のN-tuple全てに対し、評価を行うクラス
class NTupleBoard:
    def __init__(self, n: int, board_size: int, device: torch.device) -> None:
        self.n = n
        self.board_size = board_size
        self.device = device

        self.dirs = ((1, 0), (0, 1), (1, 1), (1, -1))  # 横、縦、斜め右下、斜め左下
        
        self.tuples = torch.zeros(((board_size**2)*len(self.dirs), n, 2), dtype=torch.long, device=device)
        self.lut: torch.Tensor = torch.zeros((3**n), dtype=torch.float32, device=device)  # ルックアップテーブルの初期化
        
        self.tuples_xy = torch.zeros((board_size**2, n*len(self.dirs), n, 2), dtype=torch.long, device=device)
        
        # shape: (n,)
        self.powers_of_3 = torch.pow(3, torch.arange(self.n, device=device).flip(dims=[0])).long()

        
        self.xy_idx = torch.zeros((self.board_size**2), dtype=torch.long, device=device)

        self.define_tuples()  # N-tupleの定義とルックアップテーブルの初期化
        
    
    # 全方向に対してN-tuple（座標のリスト）を定義するメソッド
    def define_tuples(self) -> None:
        idx = 0
        
        for dir in self.dirs:
            dx, dy = dir

            y_start = 0 if dy >= 0 else self.n - 1
            x_start = 0 if dx >= 0 else self.n - 1
            y_goal = self.board_size if dy <= 0 else self.board_size - self.n
            x_goal = self.board_size if dx <= 0 else self.board_size - self.n

            for y in range(y_start, y_goal):
                for x in range(x_start, x_goal):
                    # 各方向に対してN-tupleを生成
                    ntuple = make_ntuple(self.n, x, y, dx, dy, self.device)
                    
                    pos = y * self.board_size + x
                    self.tuples[(self.board_size**2)*idx + pos] = ntuple
                    
                    for i in range(self.n):
                        dpos = (y+dy*i) * self.board_size + (x+dx*i)
                        self.tuples_xy[dpos, self.xy_idx[dpos]] = ntuple
                        self.xy_idx[dpos] += 1

            idx += 1
        
    def evaluate(self, board: torch.Tensor) -> torch.Tensor:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得
        indices = self.get_lut_indices(board)
        
        # ルックアップテーブルからスコアを取得
        score = torch.sum(self.lut[indices])
        
        return score
    
    def evaluate_xy(self, board: torch.Tensor, x: int, y: int) -> torch.Tensor:
        indices = self.get_lut_indices_xy(board, x, y)
        score = torch.sum(self.lut[indices])
        
        return score
    
    def get_lut_indices_by_tuples(self, board: torch.Tensor, tuples: torch.Tensor) -> torch.Tensor:
        
        # shape: (board_size**2, n)
        x_coords = tuples[:, :, 0]
        # shape: (board_size**2, n)
        y_coords = tuples[:, :, 1]
        
        # shape: (board_size**2, n)
        values = board[y_coords, x_coords]
        
        # shape: (board_size**2,)
        indices = torch.sum(values * self.powers_of_3, dim=1)
        
        return indices
    
    def get_lut_indices(self, board: torch.Tensor) -> torch.Tensor:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得するロジックを実装
        
        return self.get_lut_indices_by_tuples(board, self.tuples)
    
    def get_lut_indices_xy(self, board: torch.Tensor, x: int, y: int) -> torch.Tensor:
        pos = y * self.board_size + x
        if(self.xy_idx[pos] >= self.n*len(self.dirs)):
            return self.get_lut_indices_by_tuples(board, self.tuples_xy[pos])
        else:
            return self.get_lut_indices_by_tuples(board, self.tuples_xy[pos][0:self.xy_idx[pos]])
        

    def update_lut(self, board: torch.Tensor, tderror: torch.Tensor, lr: float, y:torch.Tensor) -> None:
        # ルックアップテーブルの更新ロジックを実装
        indices = self.get_lut_indices(board)
        #yはモデルの出力値（tanh関数を通したスコア）
        grad = lr * tderror * (1- y**2)
        self.lut[indices] += grad

    def get_tuples(self):
        return self.tuples

    def get_lut(self):
        return self.lut



# 五目並べのN-tupleネットワークの実装
class NTupleNetwork:
    def __init__(self, board_size: int, device: torch.device,  learning_rate: float = 0.01) -> None:
        self.board_size = board_size
        self.learning_rate = learning_rate
        
        self.device = device
        
        self.ev_tuple_time = 0
        self.ev_xy_time = 0
        
        self.tanh_ev_time = 0
        self.m_score_time = 0
        
        self.n_tuples: list[NTupleBoard] = []
        self.define_tuples()  # N-tupleの定義とルックアップテーブルの初期化
        
        self.m_inf = torch.tensor((-1*INF),device=device)
        
    def add_all_dir_n_tuples(self, n:int) -> None:
        self.n_tuples.append(NTupleBoard(n, self.board_size, self.device))
        
    def define_tuples(self) -> None:
        self.add_all_dir_n_tuples(10)
        #self.init_weights()  # ルックアップテーブルの初期化
        
    # 盤面の状態から選択可能な手を評価するメソッド
    def evaluate(self, board: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros((self.board_size**2), dtype=torch.float32, device=self.device)
        
        #元の状態のスコアを保存しておく
        t1 = time.time()
        base_score = 0
        for ntuple in self.n_tuples:
            base_score += ntuple.evaluate(board)
        self.ev_tuple_time += time.time() - t1
        
        for i in range(self.board_size**2):
            y = i // self.board_size
            x = i % self.board_size
            
            t1 = time.time()
            if board[y, x] != 0:
                scores[i] = self.m_inf  # すでに石が置かれている場所はスコアを-∞に設定
                continue
            self.m_score_time += time.time() - t1
            
            # 行動した結果、どのように結果が変化したか差分を取る
            t1 = time.time()
            base_value = 0
            for ntuple in self.n_tuples:
               base_value +=  ntuple.evaluate_xy(board, x, y)
            
            before_value = board[y, x]
            board[y, x] = 1
            next_value = 0
            for ntuple in self.n_tuples:
                next_value += ntuple.evaluate_xy(board, x, y)
                
            self.ev_xy_time += time.time() - t1
                
            board[y, x] = before_value  # 元の状態に戻す
            
            # スコアをtanh関数で正規化
            # ここではスコアを-1から1の範囲に収める
            t1 = time.time()
            scores[i] = torch.tanh(base_score - base_value + next_value)
            self.tanh_ev_time += time.time() - t1
        
        return scores
    
    
    def learn(self, board:torch.Tensor, tderror: torch.Tensor, y:torch.Tensor) -> None:
        for ntuple in self.n_tuples:
            ntuple.update_lut(board, tderror, self.learning_rate, y)
            
    def print_ev_times(self) -> None:
        print(f"tuple Ev time: {self.ev_tuple_time:.4f}s, xy Ev time: {self.ev_xy_time:.4f}s, tanh Ev time: {self.tanh_ev_time:.4f}s, ", end = "")
        print(f"mscore in time: {self.m_score_time:4f}s")
        
    def reset_ev_times(self) -> None:
        self.ev_tuple_time = 0
        self.ev_xy_time = 0
        self.tanh_ev_time = 0
            
            