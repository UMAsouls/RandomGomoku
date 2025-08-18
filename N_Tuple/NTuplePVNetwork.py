from N_Tuple import NTupleNetwork
from N_Tuple.NTupleNetwork import NTupleBoard

import numpy as np

INF = 100000000

class NTuplePVBoard(NTupleBoard):
    def __init__(self, n: int, board_size: int, p_dim: int) -> None:
        super().__init__(n,board_size)
        
        self.lut: np.ndarray = np.zeros((3**n, 1+p_dim), dtype=np.float64)
        
    def evaluatePV(self, board) -> tuple[np.ndarray, float]:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得
        indices = self.get_lut_indices(board)
        
        # ルックアップテーブルからスコアを取得
        scores:np.ndarray = np.sum(self.lut[indices], axis=0)
        
        value = scores[0]
        policies = scores[1:]
        
        return policies, value
        
    
    def learnPV(self, board: np.ndarray, p_costs: np.ndarray, v_cost: float, lr: float):
        indices = self.get_lut_indices(board)

        costs = np.insert(p_costs, 0, v_cost)
        
        self.lut[indices]-= lr * costs
        
        

class NTuplePVNetwork(NTupleNetwork):
    def __init__(self, board_size, ts = ..., learning_rate = 0.01):
        super().__init__(board_size, ts, learning_rate)
        
        self.n_tuples: list[NTuplePVBoard] = []
        self.define_tuples()
        
    def add_all_dir_n_tuples(self, n:int) -> None:
        self.n_tuples.append(NTuplePVBoard(n, self.board_size, self.board_size**2))
        
    def evaluatePV(self, board: np.ndarray) -> tuple[np.ndarray, float]:
        policies = np.zeros(self.board_size**2, dtype=np.float64)
        value = 0
        
        for i in self.n_tuples:
            p,v = i.evaluatePV(board)
            policies += p
            value += v
            
        return policies, np.tanh(value)
    
    def learnPV(self, board: np.ndarray, p_costs: np.ndarray, v_cost: float, v_raw: float = -INF):
        v_y: float
        if(v_raw == -INF): _, v_y = self.evaluatePV(board)
        else: v_y = v_raw
        
        v_cost = v_cost*(1-v_y**2)
        
        for i in self.n_tuples:
            i.learnPV(board, p_costs, v_cost, self.learning_rate)
        
    