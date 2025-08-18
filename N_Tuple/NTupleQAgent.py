import numpy as np

from N_Tuple.NTupleNetwork import NTupleNetwork


BATCH_SIZE = 32
BUFFER_SIZE = 10000
LEARNING_RATE = 0.01
EPSILON = 0.01  # ε-greedy法のε値
GAMMA = 0.9


class NTupleQAgent:
    def __init__(self, board_size=19, model_path = "NTupleQModel", eps = EPSILON, net_list = [10]):
        self.batch_size = BATCH_SIZE
        self.buffer_size = BUFFER_SIZE
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        self.epsilon = eps
        self.ntuple_network = NTupleNetwork(board_size=board_size, ts = net_list, learning_rate=self.learning_rate)

        self.board_size = board_size

        self.model_path = model_path
        
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
            # N-tupleネットワークを使用して最適な手を選ぶ
            scores = self.ntuple_network.evaluate(board)
            
            max_value = np.max(scores)
            indices = np.where(scores == max_value)[0]  # 最大値のインデックスを取得
            action = np.random.choice(indices)
            return action
        
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.get_batch()
        """
        qs = self.ntuple_network.evaluate(state)
        
        # next_stateは相手から見た盤面 → マイナスする
        tdtarget = reward - (1 - done) * self.gamma * np.max(self.ntuple_network.evaluate(next_state))
        tderror = qs[action] - tdtarget
        
        self.ntuple_network.learn(state, action, tderror, qs[action])

        return tderror**2

        
        
        # N-tupleネットワークの学習

    def save(self):
        self.ntuple_network.save(self.model_path)

    def load(self):
        self.ntuple_network.load(self.model_path)
        
            