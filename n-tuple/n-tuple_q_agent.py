import numpy as np

from NTupleNetwork import NTupleNetwork
from ReplayBuffer import ReplayBuffer


BATCH_SIZE = 32
BUFFER_SIZE = 10000
LEARNING_RATE = 0.0005
EPSILON = 0.1  # ε-greedy法のε値


class NTupleQAgent:
    def __init__(self, board_size=19):
        self.batch_size = BATCH_SIZE
        self.buffer_size = BUFFER_SIZE
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.ntuple_network = NTupleNetwork(board_size=board_size, learning_rate=self.learning_rate)
        
    def select_action(self, board: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            # ランダムに手を選ぶ
            return np.random.randint(0, board.size)
        else:
            # N-tupleネットワークを使用して最適な手を選ぶ
            scores = self.ntuple_network.forward(board)
            return np.argmax(scores)
        
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.get_batch()
        
        # N-tupleネットワークの学習
        for exp in batch:
            tderror = exp.reward + (1 - exp.done) * np.max(self.ntuple_network.forward(exp.next_state)) - \
                      self.ntuple_network.forward(exp.state)[exp.action]
            self.ntuple_network.learn(exp.state, tderror)