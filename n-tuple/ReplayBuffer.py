import numpy as np
from dataclasses import dataclass


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        self.buffer = np.zeros(self.buffer_size, dtype=object)  # オブジェクト型の配列を使用
        self.index = 0  # 現在のインデックス

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        data = Experience(state, action, reward, next_state, done)
        self.buffer[self.index] = data
        self.index = (self.index + 1) % self.buffer_size
        
    def __len__(self):
        return len(self.buffer)

    def get_batch(self) -> np.ndarray:
        batch = np.random.choice(self.buffer, self.batch_size, replace=False)
        
        return batch