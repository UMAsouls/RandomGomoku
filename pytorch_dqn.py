import copy
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import GomokuEnv
from GomokuEnv import Stone


# 経験を保存するリプレイバッファ
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self, device):
        data = random.sample(self.buffer, self.batch_size)

        def to_numpy_if_tensor(x):
            return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

        state = np.stack([to_numpy_if_tensor(x[0]) for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([to_numpy_if_tensor(x[2]) for x in data])  # ← 修正！
        next_state = np.stack([to_numpy_if_tensor(x[3]) for x in data])
        done = np.array([to_numpy_if_tensor(x[4]) for x in data]).astype(bool)  # ← 修正！

        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.bool).to(device)

        return state, action, reward, next_state, done



# Q値を出力する畳み込みニューラルネットワーク
class QNet(nn.Module):
    def __init__(self, action_size):
        super(QNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 19 * 19, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # チャンネル次元追加
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


# DQNエージェント
class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 256
        self.board_size = 19

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.board_size * self.board_size).to(self.device)
        self.qnet_target = copy.deepcopy(self.qnet).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state, player_stone):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = np.array(state)
        state = state[np.newaxis, :]  # shape: (1, 19, 19)
        state_flat = torch.tensor(state, dtype=torch.float32).to(self.device)

        qs = self.qnet(state_flat)
        qs = qs.data[0].cpu().numpy()  # GPUからCPUに戻す

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

        state, action, reward, next_state, done = self.replay_buffer.get_batch(self.device)

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
        self.qnet.load_state_dict(torch.load(os.path.join(path, 'qnet.pth'), map_location=self.device))
        self.qnet.to(self.device)
        print(f"Model loaded from {path}")
