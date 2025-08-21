from N_Tuple import NTupleNetwork
from N_Tuple.NTupleNetwork import NTupleBoard

import numpy as np

INF = 100000000

class NTuplePVBoard(NTupleBoard):
    def __init__(self, n: int, board_size: int, p_dim: int) -> None:
        super().__init__(n,board_size)
        
        self.lut: np.ndarray = np.zeros((3**n, 1+1+self.n), dtype=np.float64)
        
    def evaluatePV(self, board) -> tuple[np.ndarray, float]:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得
        indices = self.get_lut_indices(board)
        
        xy_indices = self.get_lut_indices_by_xy_tuples(board)
        
        # ルックアップテーブルからスコアを取得
        value = np.sum(self.lut[:,0][indices])
        
        base_p = np.sum(self.lut[:,1][indices])
        policies = np.full(self.board_size**2, base_p, dtype=np.float64)
        
        bef_ps = self.lut[xy_indices, 1]
        aft_ps = self.lut[xy_indices, 2+self.xy_tuples_pos]
        
        delta_ps = np.where(self.xy_tuples_mask, aft_ps-bef_ps, 0)
        
        delta = np.sum(delta_ps, axis=1)
        
        policies += delta
        
        return policies, value
        
    
    def learnPV(self, board: np.ndarray, p_costs: np.ndarray, v_cost: float, lr: float):
        indices = self.get_lut_indices(board)
        
        xy_indices = self.get_lut_indices_by_xy_tuples(board)
        
        self.lut[indices,0]-= lr * v_cost
        
        self.lut[indices,1] -= lr * np.sum(p_costs)
        
        ps = np.repeat(lr*p_costs[:,np.newaxis], xy_indices.shape[1], axis=-1)
        ps = np.where(self.xy_tuples_mask, ps, 0)
        self.lut[xy_indices, 1] += ps
        self.lut[xy_indices, 2+self.xy_tuples_pos] -= ps
        
        
        

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
        
    