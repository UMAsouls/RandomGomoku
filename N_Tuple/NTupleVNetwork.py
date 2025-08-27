from N_Tuple import NTupleNetwork
from N_Tuple.NTupleNetwork import NTupleBoard

import numpy as np

INF = 100000000


class NTupleVBoard(NTupleBoard):
    def __init__(self, n: int, board_size: int, p_dim: int) -> None:
        super().__init__(n,board_size)
        
    def evaluatePV(self, board) -> tuple[np.ndarray, float]:
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得
        indices = self.get_lut_indices(board)
        value = np.sum(self.lut[indices])
        
        uy,ux = np.where(board!=0)
        unlegal = uy*self.board_size + ux
        
        policies = np.full((self.board_size**2), value, dtype=np.float64)
        
        xy_indices = self.get_lut_indices_by_xy_tuples(board)
        
        indices_dif = (3**(self.n-self.xy_tuples_pos-1))
        indices_dif[unlegal] = np.zeros(xy_indices.shape[1], dtype=np.int64)
        indices_dif = np.where(self.xy_tuples_mask, indices_dif, 0)
        
        aft_indices = xy_indices+indices_dif
        
        bef_ps = self.lut[xy_indices]
        aft_ps = self.lut[aft_indices]
        
        delta_ps = np.where(self.xy_tuples_mask, aft_ps-bef_ps, 0)
        delta = np.sum(delta_ps, axis=1)
        
        policies += delta
        policies[unlegal] = -INF
        
        return policies, value
    
    
    def learnPV(self, board: np.ndarray, p_costs: np.ndarray, v_cost: float, p_lr: float, v_lr: float):
        indices = self.get_lut_indices(board)
        
        self.lut[indices]-= v_lr * v_cost / len(indices)
        
class NTupleVNetwork(NTupleNetwork):
    def __init__(self, board_size, ts = ..., p_lr = 0.01, v_lr = 0.001):
        super().__init__(board_size, ts, 0.01)
        
        self.n_tuples: list[NTupleVBoard] = []
        self.define_tuples()
        
        self.p_lr = p_lr
        self.v_lr = v_lr
        
    def add_all_dir_n_tuples(self, n:int) -> None:
        self.n_tuples.append(NTupleVBoard(n, self.board_size, self.board_size**2))
        
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
            i.learnPV(board, p_costs, v_cost, self.p_lr, self.v_lr)
            
        