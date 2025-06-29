import numpy as np
import torch

import time

from N_Tuple.NTupleNetwork import NTupleNetwork


BATCH_SIZE = 32
BUFFER_SIZE = 10000
LEARNING_RATE = 0.01
EPSILON = 0.01  # ε-greedy法のε値
GAMMA = 0.9

# GPUが利用可能かチェックし、利用可能なら 'cuda' を、そうでなければ 'cpu' を設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NTupleQAgent:
    def __init__(self, board_size=19):
        self.batch_size = BATCH_SIZE
        self.buffer_size = BUFFER_SIZE
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.device = device
        
        self.ntuple_network = NTupleNetwork(board_size, device, learning_rate=self.learning_rate)
        
        self.ev_time = 0
        self.tdtarget_ev_time = 0
        
    def random_empty_action(self, board: np.ndarray) -> int:
        # ランダムに手を選ぶ
        # 空いている場所をランダムに選ぶ
        
        
        empty_indices = np.where(board == 0)
        empty = np.array(empty_indices).T  # 空いている場所の座標を取得
        if len(empty) == 0:
            raise ValueError("空いている場所がありません")
        pos = empty[np.random.choice(len(empty))]  # ランダムに1つ選ぶ
        action = pos[0] * board.shape[1] + pos[1]
        return action
        
    def select_action(self, board: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return self.random_empty_action(board)
            
        else:
            board_tensor = torch.from_numpy(board).to(device=self.device)
            
            # N-tupleネットワークを使用して最適な手を選ぶ
            t1 = time.time()
            scores = self.ntuple_network.evaluate(board_tensor)
            self.ev_time += time.time() - t1
            
            max_value = torch.max(scores)
            if max_value <= 0:
                return self.random_empty_action(board)
            indices = torch.where(scores == max_value)[0]  # 最大値のインデックスを取得
            
            # 全ての要素が同じ確率で選ばれるように、重みを全て1にする
            weights_uniform = torch.ones(len(indices))
            idx = torch.multinomial(weights_uniform, 1, replacement=False)

            action = indices[idx].item()
            return action
        
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.get_batch()
        """
        state_tensor = torch.from_numpy(state).to(device=self.device)
        next_state_tensor = torch.from_numpy(next_state).to(device=self.device)
        
        t1 = time.time()
        q = self.ntuple_network.evaluate(state_tensor)
        self.ev_time += time.time() - t1
        
        t1 = time.time()
        tdtarget = reward + (1 - done) * self.gamma * torch.max(self.ntuple_network.evaluate(next_state_tensor)).item()
        self.ev_time += time.time() - t1
        self.tdtarget_ev_time += time.time() - t1
        tderror = tdtarget - q[action]
        
        self.ntuple_network.learn(state_tensor, tderror, q[action])
        
        
        # N-tupleネットワークの学習
        
    def print_net_ev_times(self) -> None:
        self.ntuple_network.print_ev_times()
        self.ntuple_network.reset_ev_times()
        
    def print_ev_times(self) -> None:
        print(f"All Evaluate Time: {self.ev_time:.4f}s, TDtarget Ev time: {self.tdtarget_ev_time:.4f}s")
        self.reset_ev_times()
    
    def reset_ev_times(self) -> None:
        self.ev_time = 0
        self.tdtarget_ev_time = 0
            