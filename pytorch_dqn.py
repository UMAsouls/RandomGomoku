import copy
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import math
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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
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


# モンテカルロ木探索のノードクラス
class MCTSNode:
    def __init__(self, state, parent=None, action=None, player_stone=None):
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.player_stone = player_stone
        
    def add_child(self, state, action, player_stone):
        child = MCTSNode(state, self, action, player_stone)
        self.children.append(child)
        return child
        
    def update(self, result):
        self.visits += 1
        self.value += result
        
    def fully_expanded(self, valid_actions):
        return len(self.children) == len(valid_actions)
    
    def best_child(self, c_param=1.4):
        # UCB1アルゴリズムに基づく最良の子ノード選択
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            # 探索と活用のバランスを取るUCB1スコア計算
            exploit = child.value / child.visits if child.visits > 0 else 0
            explore = c_param * math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

# モンテカルロ木探索エージェント
class MCTSAgent:
    def __init__(self, simulations=10000, board_size=19):
        self.simulations = simulations
        self.board_size = board_size
        
    def get_action(self, state, player_stone):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        
        root = MCTSNode(state, player_stone=player_stone)
        
        for _ in range(self.simulations):
            node = self.select(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        
        # 最も訪問回数が多い子ノードの行動を選択
        if not root.children:
            # 有効な行動がない場合はランダムに選択
            valid_actions = self.get_valid_actions(state)
            if valid_actions:
                return random.choice(valid_actions)
            return (0, 0)  # フォールバック

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def select(self, node):
        # 選択フェーズ: UCBに基づいて最良のノードを選択
        current = node
        while current.children:
            if all(child.visits > 0 for child in current.children):
                current = current.best_child()
            else:
                # 未訪問の子ノードがある場合はそれを選択
                unexplored = [child for child in current.children if child.visits == 0]
                return random.choice(unexplored)
        
        # 選択したノードを展開
        valid_actions = self.get_valid_actions(current.state)
        if valid_actions and not current.fully_expanded(valid_actions):
            return self.expand(current, valid_actions)
        
        return current
    
    def expand(self, node, valid_actions):
        # 展開フェーズ: 新しい子ノードを追加
        taken_actions = [child.action for child in node.children]
        possible_actions = [action for action in valid_actions if action not in taken_actions]
        
        if not possible_actions:
            return node
            
        action = random.choice(possible_actions)
        x, y = action
        new_state = copy.deepcopy(node.state)
        next_player = 3 - node.player_stone  # 相手の石
        new_state[y][x] = node.player_stone
        child = node.add_child(new_state, action, next_player)
        return child
    
    def simulate(self, node):
        # シミュレーションフェーズ: ランダムプレイアウト
        state = copy.deepcopy(node.state)
        current_player = node.player_stone
        original_player = node.player_stone
        
        while True:
            valid_actions = self.get_valid_actions(state)
            if not valid_actions:
                return 0  # 引き分け
            
            action = random.choice(valid_actions)
            x, y = action
            state[y][x] = current_player
            
            # 勝敗チェック
            if self.check_win(state, current_player, (y, x)):
                return 1 if current_player == original_player else -1
            
            current_player = 3 - current_player
    
    def backpropagate(self, node, reward):
        # バックプロパゲーションフェーズ: 結果を木に反映
        while node:
            node.update(reward)
            node = node.parent
            reward = -reward  # 親と子で価値を反転
    
    def get_valid_actions(self, state):
        # 有効な行動（空いているマス）を取得
        valid_actions = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if state[y][x] == 0:
                    valid_actions.append((x, y))
        return valid_actions
    
    def check_win(self, state, player_stone, last_move):
        # 勝利判定（五目並べのルールに従って）
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、縦、右下がり、右上がり
        y, x = last_move
        
        for dy, dx in directions:
            count = 1  # 自分自身をカウント
            
            # 正方向に探索
            for i in range(1, 5):
                ny, nx = y + i * dy, x + i * dx
                if 0 <= ny < self.board_size and 0 <= nx < self.board_size and state[ny][nx] == player_stone:
                    count += 1
                else:
                    break
            
            # 逆方向に探索
            for i in range(1, 5):
                ny, nx = y - i * dy, x - i * dx
                if 0 <= ny < self.board_size and 0 <= nx < self.board_size and state[ny][nx] == player_stone:
                    count += 1
                else:
                    break
            
            if count >= 5:
                return True
        
        return False
