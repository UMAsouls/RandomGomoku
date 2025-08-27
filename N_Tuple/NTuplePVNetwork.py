from N_Tuple import NTupleNetwork
from N_Tuple.NTupleNetwork import NTupleBoard

import numpy as np

INF = 100000000

class NTuplePVBoard(NTupleBoard):
    def __init__(self, n: int, board_size: int) -> None:
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
        
    
    def learnPV(self, board: np.ndarray, p_costs: np.ndarray, v_cost: float, p_lr: float, v_lr: float):
        indices = self.get_lut_indices(board)
        
        xy_indices = self.get_lut_indices_by_xy_tuples(board)
        
        self.lut[indices,0]-= v_lr * v_cost / len(indices)
        
        self.lut[indices,1] -= p_lr * np.sum(p_costs)
        
        ps = np.repeat(p_lr * p_costs[:,np.newaxis], xy_indices.shape[1], axis=-1)
        ps = np.where(self.xy_tuples_mask, ps, 0)
        self.lut[xy_indices, 1] += ps
        self.lut[xy_indices, 2+self.xy_tuples_pos] -= ps
        

class NTuplePVBoard2(NTuplePVBoard):
    def __init__(self, n: int, board_size: int) -> None:
        super().__init__(n,board_size)
        
        self.lut: np.ndarray = np.zeros((3**n, 2), dtype=np.float64)
        
    def evaluatePV(self, board):
        # ボードの状態に基づいてルックアップテーブルのインデックスを取得
        indices = self.get_lut_indices(board)
        
        xy_indices = self.get_lut_indices_by_xy_tuples(board)
        
        # ルックアップテーブルからスコアを取得
        value = np.sum(self.lut[indices, 0])
        
        uy,ux = np.where(board!=0)
        unlegal = uy*self.board_size + ux
        
        base_p = np.sum(self.lut[indices,1])
        policies = np.full(self.board_size**2, base_p, dtype=np.float64)
        
        aft_indices = self.get_aft_indices(xy_indices, unlegal)
        
        bef_ps = self.lut[xy_indices, 1]
        aft_ps = self.lut[aft_indices, 1]
        
        delta_ps = np.where(self.xy_tuples_mask, aft_ps-bef_ps, 0)
        delta = np.sum(delta_ps, axis=1)
        
        policies += delta
        policies[unlegal] = -INF
        
        return policies, value
    
    def learnPV(self, board: np.ndarray, p_costs: np.ndarray, v_cost: float, p_lr: float, v_lr: float):
        indices = self.get_lut_indices(board)
        
        xy_indices = self.get_lut_indices_by_xy_tuples(board)
        
        self.lut[indices,0]-= v_lr * v_cost / len(indices)
        self.lut[indices,1] -= p_lr * np.sum(p_costs)
        
        uy,ux = np.where(board!=0)
        unlegal = uy*self.board_size + ux
        aft_indices = self.get_aft_indices(xy_indices, unlegal)
        
        ps = np.repeat(p_lr * p_costs[:,np.newaxis], xy_indices.shape[1], axis=-1)
        ps = np.where(self.xy_tuples_mask, ps, 0)
        self.lut[xy_indices, 1] += ps
        self.lut[aft_indices, 1] -= ps
        

class NTuplePVNetwork(NTupleNetwork):
    def __init__(self, board_size, ts = ...,  p_lr = 0.01, v_lr = 0.001, use_v2 = False):
        self.use_v2 = use_v2
        super().__init__(board_size, ts, 0.1)
        
        self.n_tuples: list[NTuplePVBoard] = []
        self.define_tuples()
        
        self.p_lr = p_lr
        self.v_lr = v_lr
        
    def add_all_dir_n_tuples(self, n:int) -> None:
        if(not self.use_v2): self.n_tuples.append(NTuplePVBoard(n, self.board_size))
        else: self.n_tuples.append(NTuplePVBoard2(n, self.board_size))
        
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
        
    