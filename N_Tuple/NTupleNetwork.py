import numpy as np
import collections

from N_Tuple.ReplayBuffer import Experience

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
                    ntuple = make_ntuple(self.n, x, y, dx, dy)
                    self.tuples[(self.board_size**2)*idx + (y * self.board_size + x)] = ntuple

            idx += 1
        
    def evaluate(self, board: np.ndarray) -> float:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得
        indices = self.get_lut_indices(board)
        
        # ルックアップテーブルからスコアを取得
        score = np.sum(self.lut[indices])
        
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

    def update_lut(self, board: np.ndarray, tderror: float, lr: float, y:float) -> None:
        # ルックアップテーブルの更新ロジックを実装
        indices = self.get_lut_indices(board)
        #yはモデルの出力値（tanh関数を通したスコア）
        self.lut[indices] += lr * tderror * (1- y**2)    

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
        self.define_tuples()  # N-tupleの定義とルックアップテーブルの初期化
        
    def init_weights(self, mu: float = 0, sigma: float = 1) -> None:
        # ルックアップテーブルの初期化
        rng = np.random.default_rng()
        for ntuple in self.n_tuples:
            ntuple.lut = rng.normal(mu, sigma, ntuple.lut.shape)
        
    def add_all_dir_n_tuples(self, n:int) -> None:
        self.n_tuples.append(NTupleBoard(n, self.board_size))
        
    def define_tuples(self) -> None:
        self.add_all_dir_n_tuples(5)
        #self.init_weights()  # ルックアップテーブルの初期化
        
    # 盤面の状態から選択可能な手を評価するメソッド
    def evaluate(self, board: np.ndarray) -> np.ndarray:
        scores = np.zeros((self.board_size**2), dtype=np.float64)
        for i in range(self.board_size**2):
            y = i // self.board_size
            x = i % self.board_size
            
            if board[y, x] != 0:
                scores[i] = -1*INF  # すでに石が置かれている場所はスコアを-∞に設定
                continue
            
            before_value = board[y, x]
            board[y, x] = 1
            # 空いている場所に対してN-tupleを評価
            score = 0
            for ntuple in self.n_tuples:
                score += ntuple.evaluate(board)
                
            board[y, x] = before_value  # 元の状態に戻す
            
            
            # スコアをtanh関数で正規化
            # ここではスコアを-1から1の範囲に収める
            scores[i] = np.tanh(score)
            
        return scores
    
    
    def learn(self, board:np.ndarray, tderror: float, y:float) -> None:
        for ntuple in self.n_tuples:
            ntuple.update_lut(board, tderror, self.learning_rate, y)
            
            