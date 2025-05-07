import copy
from collections import deque
import random
import numpy as np
import os
import dezero
import dezero.functions as F
import dezero.layers as L
from dezero import Model
from dezero import optimizers

#   import gym #   Not used

#   import GomokuEnv
#   from GomokuEnv import Stone #   Not used
#   赤＝黒：青＝白 #   Irrelevant comment


# Feature Encoding
def encode_state(state, player_stone):
    """Encodes the Gomoku board state into two 15x15 feature matrices."""

    encoded_state = np.zeros((2, 15, 15), dtype=np.float32)  # Initialize as float32

    if player_stone == 1:  # Assuming 1 for black
        my_stone = 1
        opponent_stone = 2
    else:
        my_stone = 2
        opponent_stone = 1

    for i in range(15):
        for j in range(15):
            if state[i][j] == my_stone:
                encoded_state[0][i][j] = 1
            elif state[i][j] == opponent_stone:
                encoded_state[1][i][j] = 1

    return encoded_state


# SE-ResNet Module
class SE_ResNet_Block(Model):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv1 = L.Conv2d(channel, channel, 3, pad=1)
        self.squeeze = L.Linear(channel, channel // reduction)
        self.excitation = L.Linear(channel // reduction, channel)

    def forward(self, x):
        residual = x
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv1(h))  #   Second Conv
        # Squeeze
        s = F.average_pooling(h, h.shape[2])
        s = F.relu(self.squeeze(s))
        # Excitation
        e = F.sigmoid(self.excitation(s))
        h = h * e
        return h + residual


# Q-Network with SE-ResNet Modules
class GomokuQNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = L.Conv2d(2, 16, 3, pad=1)  # Input channels = 2
        self.res1 = SE_ResNet_Block(16)
        self.res2 = SE_ResNet_Block(16)
        self.res3 = SE_ResNet_Block(16)
        self.res4 = SE_ResNet_Block(16)
        self.res5 = SE_ResNet_Block(16)
        self.res6 = SE_ResNet_Block(16)
        self.fc1 = L.Linear(16 * 15 * 15, 225)
        self.fc2 = L.Linear(225, action_size)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = h.reshape(x.shape[0], -1)  # Flatten
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h


# Replay Buffer (same as before)
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(bool)
        return state, action, reward, next_state, done


class DQNAgent:
    def __init__(self, board_size=15):  # Gomoku is 15x15
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = GomokuQNet(self.action_size)  # Use GomokuQNet
        self.qnet_target = GomokuQNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

        self.player_stone = 1  # Or 2, set at the beginning of the game
        self.elo_score = 1500  # Initial Elo

    def get_action(self, state, valid_moves):
        """
        Gets an action (move) from the Q-network, following an epsilon-greedy policy.

        Args:
            state:  The current board state (15x15 numpy array).
            valid_moves: A list of (row, col) tuples representing legal moves.

        Returns:
            A tuple (row, col) representing the chosen action.
        """
        encoded_state = encode_state(state, self.player_stone)  # Encode state
        encoded_state = encoded_state[np.newaxis, :]  # (1, 2, 15, 15)

        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random valid move
            return random.choice(valid_moves)
        else:
            # Exploitation: Choose the move with the highest Q-value
            qs = self.qnet(encoded_state)
            qs = qs.data[0]  # (1, 225) -> (225,)

            # Get Q-values for valid moves
            valid_move_indices = [r * self.board_size + c for r, c in valid_moves]
            valid_q_values = qs[valid_move_indices]

            best_move_index = np.argmax(valid_q_values)
            return valid_moves[best_move_index]

    def update(self, state, action, reward, next_state, done):
        encoded_state = encode_state(state, self.player_stone)
        encoded_next_state = encode_state(next_state, self.player_stone)

        self.replay_buffer.add(encoded_state, self.action_to_index(action), reward, encoded_next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.get_batch()
        self.learn(batch)

    def learn(self, batch):
        state, action, reward, next_state, done = batch

        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = F.max(next_qs, axis=1)
        target = reward + (1 - done) * self.gamma * next_q
        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def action_to_index(self, action):
        """Converts (row, col) action to an index."""
        return action[0] * self.board_size + action[1]

    def index_to_action(self, index):
        """Converts an index to (row, col) action."""
        return index // self.board_size, index % self.board_size

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.qnet.save_weights(os.path.join(path, "qnet.npz"))
        print(f"Model saved at {path}")

    def load(self, path):
        self.qnet.load_weights(os.path.join(path, "qnet.npz"))
        print(f"Model loaded from {path}")


#   Example Usage (Illustrative)
if __name__ == "__main__":
    agent = DQNAgent()
    #   env = GomokuEnv()
    state = np.zeros((15, 15), dtype=np.int32)  #   Example initial state
    done = False
    player_stone = 1
    valid_moves = [(r, c) for r in range(15) for c in range(15)]  #   All positions initially

    for step in range(1000):  #   Example training loop
        action = agent.get_action(state, valid_moves)
        print(f"Action: {action}")
        #   next_state, reward, done, _ = env.step(action)
        next_state = np.zeros((15, 15), dtype=np.int32)
        reward = 0
        done = False
        #   state = next_state
        agent.update(state, action, reward, next_state, done)
        if done:
            state = np.zeros((15, 15), dtype=np.int32)
            done = False
        state = next_state
        if step % 100 == 0:
            agent.sync_qnet()
    agent.save("gomoku_agent")