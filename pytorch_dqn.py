import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import GomokuEnv
from GomokuEnv import Stone

# DQNの経験を保存するバッファ
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

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)

        return state, action, reward, next_state, done

# 盤面の状態を入力し、各手のQ値を出力するニューラルネットワーク
class QNet(nn.Module):
    def __init__(self, action_size):
        super(QNet, self).__init__()
        self.l1 = nn.Linear(19 * 19, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 128
        self.board_size = 19
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.board_size * self.board_size)
        self.qnet_target = copy.deepcopy(self.qnet)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state, player_stone):
        state = np.array(state)
        state = state[np.newaxis, :]
        state_flat = torch.tensor(state, dtype=torch.float32)

        qs = self.qnet(state_flat)
        qs = qs.data[0]

        empty_positions = [(j, i) for i in range(self.board_size) for j in range(self.board_size) if state[0][i][j] == 0]
        empty_q_values = [qs[i * self.board_size + j] for j, i in empty_positions]

        if np.random.rand() < self.epsilon:
            action = random.choice(empty_positions)
        else:
            max_q_index = np.argmax(empty_q_values)
            action = empty_positions[max_q_index]

        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        qs = self.qnet(state)
        # Convert action (x, y) to a single index
        action_indices = action[:, 1] * self.board_size + action[:, 0]
        q = qs.gather(1, action_indices.view(-1, 1)).squeeze()

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        target = reward + (1 - done.float()) * self.gamma * next_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.qnet.state_dict(), os.path.join(path, 'qnet.pth'))
        print(f"Model saved at {path}")

    def load(self, path):
        self.qnet.load_state_dict(torch.load(os.path.join(path, 'qnet.pth')))
        print(f"Model loaded from {path}")