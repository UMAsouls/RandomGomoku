import numpy as np
import collections

from ReplayBuffer import Experience

# N-tupleの取り出し座標を作成
def make_ntuple(self, n: int, x: int, y: int, dx: int, dy: int) -> np.ndarray:
    ntuple = np.zeros((n, 2), dtype=np.int32)
    for i in range(n):
        ntuple[i] = (x + i * dx, y + i * dy)
        
    return ntuple

# 盤面に存在しうる1方向のN-tuple全てに対し、評価を行うクラス
class NTupleBoard:
    def __init__(self, n: int, board_size: int, dir: tuple[int,int]) -> None:
        self.n = n
        self.board_size = board_size
        
        self.tuples = np.zeros((board_size**2), dtype=object)
        self.lut = np.zeros((board_size**2, 3**n), dtype=np.float64)  # ルックアップテーブルの初期化
        
        self.define_tuples(dir)
    
    # 指定された方向に対してN-tuple（座標のリスト）を定義するメソッド
    def define_tuples(self, dir: tuple[int,int]) -> None:
        dx, dy = dir
        
        y_start = 0 if dy >= 0 else self.n - 1
        x_start = 0 if dx >= 0 else self.n - 1
        y_goal = self.board_size if dy <= 0 else self.board_size - self.n
        x_goal = self.board_size if dx <= 0 else self.board_size - self.n
        
        for y in range(y_start, y_goal):
            for x in range(x_start, x_goal):
                # 各方向に対してN-tupleを生成
                ntuple = make_ntuple(self.n, x, y, dx, dy)
                self.tuples[y * self.board_size + x] = ntuple
                self.lut[y * self.board_size + x] = np.zeros((3**self.n), dtype=np.float64)
        
    def evaluate(self, board: np.ndarray) -> float:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得
        indices = self.get_lut_indices(board)
        
        # ルックアップテーブルからスコアを取得
        score = np.sum(self.lut[np.arange(self.board_size**2), indices])
        
        return score
    
    def get_lut_indices(self, board: np.ndarray) -> np.ndarray:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得するロジックを実装
        
        # shape: (board_size**2, n)
        x_coords = self.tuples[:, :, 0]
        # shape: (board_size**2, n)
        y_coords = self.tuples[:, :, 1]
        
        # shape: (board_size**2, n)
        values = board[y_coords, x_coords]
        
        # shape: (n,)
        powers_of_3 = np.power(3, np.arange(self.n)[::-1])
        
        # shape: (board_size**2,)
        indices = np.sum(values * powers_of_3, axis=1)
        
        return indices

    def update_lut(self):
        # ルックアップテーブルの更新ロジックを実装
        pass

    def get_tuples(self):
        return self.tuples

    def get_lut(self):
        return self.lut



# 五目並べのN-tupleネットワークの実装
class NTupleNetwork:
    def __init__(self, board_size: int, learning_rate: float = 0.01) -> None:
        self.board_size = board_size
        self.learning_rate = learning_rate
        
        self.n_tuples: list[NTupleBoard] = []
        
    def define_all_dir_n_tuples(self, n:int) -> None:
        self.n_tuples: list[NTupleBoard] = [
            NTupleBoard(n, self.board_size, (1, 0)),  # 横方向
            NTupleBoard(n, self.board_size, (0, 1)),  # 縦方向
            NTupleBoard(n, self.board_size, (1, 1)),  # 斜め右下方向
            NTupleBoard(n, self.board_size, (1, -1))  # 斜め左下方向
        ]
        
    def define_tuples(self) -> None:
        self.define_all_dir_n_tuples(5)
        
    # 盤面の状態から選択可能な手を評価するメソッド
    def forward(self, board: np.ndarray) -> np.ndarray:
        scores = np.zeros((self.board_size**2), dtype=np.float64)
        for i in range(self.board_size**2):
            if board[i // self.board_size, i % self.board_size] != 0:
                scores[i] = -1
            
            board_copy = board.copy()
            board_copy[i // self.board_size, i % self.board_size] = 1
            # 空いている場所に対してN-tupleを評価
            score = 0
            for ntuple in self.n_tuples:
                score += ntuple.evaluate(board_copy)
            
            scores[i] = score
            
        return scores
    
    
    def learn(self, board:np.ndarray, tderror: float) -> None:
        for ntuple in self.n_tuples:
            indices = ntuple.get_lut_indices(board)
            ntuple.lut[np.arange(ntuple.board_size**2), indices] += self.learning_rate * tderror
            
