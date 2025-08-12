import numpy as np
from dataclasses import dataclass

class MCTSReplayBuffer:
    def __init__(self, buffer_size, batch_size, board_size) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        experience = np.dtype([
            ("state", np.int64, (board_size, board_size)),
            ("policies", np.float64, (board_size**2)),
            ("value", np.float64)
        ])
        
        self.buffer = np.zeros(self.buffer_size, dtype= experience)
        self.index = 0  # 現在のインデックス
        self.size = 0

    def add(self, state: np.ndarray, action: int, policy: np.ndarray, value: int) -> None:
        self.buffer["state"][self.index] = state
        self.buffer["action"][self.index] = action
        self.buffer["policies"][self.index] = policy
        self.buffer["value"][self.index] = value
        
        self.index = (self.index + 1) % self.buffer_size
        
        if(self.size < self.buffer_size): self.size += 1
        
    def __len__(self) -> int:
        return self.size

    def get_batch(self) -> np.ndarray:
        buf = self.buffer[:self.size] if self.size < self.buffer_size else self.buffer  # 有効な部分だけを取得
        
        indices = np.random.choice(self.size, self.batch_size, replace=False)
        
        return self.buffer[indices]
    
    
if __name__ == "__main__":
    buf = MCTSReplayBuffer(3,1,9)
    
    buf.add(np.zeros((9,9)), 9, np.zeros(9*9), 2)
    buf.add(np.ones((9,9)), 8, np.ones(9*9), 3)
    
    print(buf.get_batch()["value"])