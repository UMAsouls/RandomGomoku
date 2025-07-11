"""
リプレイバッファ
"""

import numpy as np
from collections import deque


class ReplayBuffer:
    """経験リプレイバッファ"""
    
    def __init__(self, capacity=40000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, policy, value):
        """状態、方策、価値のタプルを追加"""
        self.buffer.append((state, policy, value))
        
    def sample(self, batch_size):
        """ランダムにバッチサイズ分のサンプルを取得"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, policies, values = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(policies), np.array(values)
    
    def __len__(self):
        return len(self.buffer)
