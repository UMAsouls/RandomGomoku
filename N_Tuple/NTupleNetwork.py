import numpy as np
import collections

import os

INF = 100

# N-tupleの取り出し座標を作成
def make_ntuple(n: int, x: int, y: int, dx: int, dy: int) -> np.ndarray:
    ntuple = np.zeros((n, 2), dtype=np.int32)
    for i in range(n):
        ntuple[i] = (x + i * dx, y + i * dy)
        
    return ntuple

# 盤面に存在しうる縦横斜め方向のN-tuple全てに対し、評価を行うクラス
class NTupleBoard:
    def __init__(self, n: int, board_size: int) -> None:
        self.n = n
        self.board_size = board_size

        self.dirs = ((1, 0), (0, 1), (1, 1), (1, -1))  # 横、縦、斜め右下、斜め左下
        
        self.tuples = np.zeros(((board_size**2)*len(self.dirs), n, 2), dtype=np.int32)
        self.lut: np.ndarray = np.zeros((3**n), dtype=np.float64)  # ルックアップテーブルの初期化
        
        self.tuples_xy = np.zeros((board_size**2, n*len(self.dirs), n, 2), dtype=np.int32)
        
        # shape: (n,)
        self.powers_of_3 = np.power(3, np.arange(self.n)[::-1])
        
        self.xy_idx = np.zeros((self.board_size**2), dtype=np.int32)

        self.define_tuples()  # N-tupleの定義とルックアップテーブルの初期化
        
        self.tuples_flat = np.zeros(((board_size**2)*len(self.dirs), n), dtype=np.int32)
        
    
    # 全方向に対してN-tuple（座標のリスト）を定義するメソッド
    def define_tuples(self) -> None:
        idx = 0
        
        for dir in self.dirs:
            dx, dy = dir

            y_start = 0 if dy >= 0 else self.n - 1
            x_start = 0 if dx >= 0 else self.n - 1
            y_goal = self.board_size if dy <= 0 else self.board_size - (self.n - 1)
            x_goal = self.board_size if dx <= 0 else self.board_size - (self.n - 1)

            for y in range(y_start, y_goal):
                for x in range(x_start, x_goal):
                    # 各方向に対してN-tupleを生成
                    ntuple = make_ntuple(self.n, x, y, dx, dy)
                    
                    pos = y * self.board_size + x
                    self.tuples[(self.board_size**2)*idx + pos] = ntuple
                    
                    for i in range(self.n):
                        dpos = (y+dy*i) * self.board_size + (x+dx*i)
                        self.tuples_xy[dpos, self.xy_idx[dpos]] = ntuple
                        self.xy_idx[dpos] += 1

            idx += 1
            
        self.tuples_flat = self.tuples[:, :, 1] * self.board_size + self.tuples[:, :, 0]  # shape (T, n)
        
    def evaluate(self, board: np.ndarray) -> float:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得
        indices = self.get_lut_indices(board)
        
        # ルックアップテーブルからスコアを取得
        score = np.sum(self.lut[indices])
        
        return score
    
    def evaluate_xy(self, board: np.ndarray, x: int, y: int) -> float:
        indices = self.get_lut_indices_xy(board, x, y)
        score = np.sum(self.lut[indices])
        
        return score
    
    def evaluate_batch(self, boards: np.ndarray) -> float:
        num_boards = boards.shape[0]
        if num_boards == 0:
            return np.array([])
        
        
        values = np.take(boards, self.tuples_flat, axis=1)

        # 2. 各盤面の各タプルのインデックスを一括計算
        # (N, T, n) -> (N, T)
        indices_batch = np.sum(values * self.powers_of_3, axis=2)

        # 3. LUTからスコアを取得し、盤面ごとに合計する
        # (N, T) -> (N,)
        scores = np.sum(self.lut[indices_batch], axis=1)

        return scores
    
    def get_lut_indices_by_tuples(self, board: np.ndarray, tuples) -> np.ndarray:
        # shape: (board_size**2, n)
        x_coords = tuples[:, :, 0]
        # shape: (board_size**2, n)
        y_coords = tuples[:, :, 1]
        
        # shape: (board_size**2, n)
        values = board[y_coords, x_coords]
        
        # shape: (board_size**2,)
        indices = np.sum(values * self.powers_of_3, axis=1)
        
        return indices
    
    def get_lut_indices(self, board: np.ndarray) -> np.ndarray:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得するロジックを実装
        
        return self.get_lut_indices_by_tuples(board, self.tuples)
    
    def get_lut_indices_xy(self, board: np.ndarray, x: int, y: int) -> np.ndarray:
        pos = y * self.board_size + x
        if(self.xy_idx[pos] < self.n*len(self.dirs)):
            return self.get_lut_indices_by_tuples(board, self.tuples_xy[pos])
        else:
            return self.get_lut_indices_by_tuples(board, self.tuples_xy[pos][0:self.xy_idx[pos]])
        

    def update_lut(self, board: np.ndarray, tderror: float, lr: float, y:float) -> None:
        # ルックアップテーブルの更新ロジックを実装
        indices = self.get_lut_indices(board)
        #yはモデルの出力値（tanh関数を通したスコア）
        #テーブルではなく関数近似だから、更新式が少し違う
        self.lut[indices] -= lr * tderror * (1- y**2)    

    def get_tuples(self):
        return self.tuples

    def get_lut(self):
        return self.lut
    
    def save(self, path: str) -> None:
        np.save(path, self.lut)

    
    def load(self, path: str) -> None:
        self.lut = np.load(path)
        


# 五目並べのN-tupleネットワークの実装
class NTupleNetwork:
    def __init__(self, board_size: int, ts: list[int] = [10], learning_rate: float = 0.01) -> None:
        self.board_size = board_size
        self.learning_rate = learning_rate
        
        self.ts = ts
        
        self.n_tuples: list[NTupleBoard] = []
        self.define_tuples()  # N-tupleの定義とルックアップテーブルの初期化
        
    def add_all_dir_n_tuples(self, n:int) -> None:
        self.n_tuples.append(NTupleBoard(n, self.board_size))
        
    def define_tuples(self) -> None:
        for i in self.ts:
            self.add_all_dir_n_tuples(i)
            
    def get_next_boards(self, board: np.ndarray, legal_move: np.ndarray) -> np.ndarray:
        board_flat = board.ravel()
        
        num_legal = len(legal_move)
        next_flat = np.repeat(board_flat[np.newaxis, :], num_legal, axis=0)
        
        next_flat[np.arange(num_legal), legal_move] = 1
        
        return next_flat
        
    # 盤面の状態から選択可能な手を評価するメソッド
    def evaluate(self, board: np.ndarray) -> np.ndarray:
        scores = np.zeros((self.board_size**2), dtype=np.float64)
        
        legal_move = np.where(board.flatten() == 0)[0]
        
        num_legal = len(legal_move)
        if num_legal == 0:
            return np.full(self.board_size**2, -INF, dtype=np.float64)

        next_boards = self.get_next_boards(board,legal_move)
        
        next_values = np.zeros(num_legal, dtype=np.float64)
        for ntuple in self.n_tuples:
            next_values += ntuple.evaluate_batch(next_boards)
        
        scores = np.full(self.board_size**2, -INF, dtype=np.float64)
        scores[legal_move] = np.tanh(next_values)
        
        return scores
    
    
    def learn(self, board:np.ndarray, action: int, tderror: float, y:float) -> None:
        # q(s,a)はアクションを起こした後の盤面を評価したもの
        # 更新するのはアクション後の盤面でなければならない
        
        #マルチプロセスのことを考えてコピーをする
        update_state = board.copy()
        update_state[action//self.board_size, action%self.board_size] = 1

        for ntuple in self.n_tuples:
            ntuple.update_lut(update_state, tderror, self.learning_rate, y)

    def load(self, dir_path:str) -> None:
        self.ts: list[int] = []
        self.n_tuples: list[NTupleBoard] = []
        model_kind_path = dir_path + "/kind.csv"
        with open(model_kind_path, mode = "r", encoding="utf-8") as f:
            line = f.read()
            data = line.strip("\n").split(",")
            for i in data:
                self.ts.append(int(i))

        self.define_tuples()

        for i in self.n_tuples:
            i.load(dir_path + "/model" + str(i.n) + ".npy")

    def save(self, dir_path:str) -> None:
        os.makedirs(dir_path, exist_ok=True)

        model_kind_path = dir_path + "/kind.csv"
        with open(model_kind_path, mode = "w", encoding="utf-8") as f:
            data = ""
            for i in self.ts:
                data += str(i) + ","

            f.write(data[:-1])

        for i in self.n_tuples:
            i.save(dir_path + "/model" + str(i.n) + ".npy")
            
            