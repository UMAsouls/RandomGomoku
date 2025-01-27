import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
import os
import GomokuEnv
from GomokuEnv import Stone

class RandomAgent:
    def __init__(self):
        pass

    def get_action(self, state):
        empty_positions = [(i, j) for i in range(19) for j in range(19) if state[i][j] == 0]
        return random.choice(empty_positions)

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


class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.board_size = 19  # 盤面のサイズ
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.board_size * self.board_size)  # 空いているマスの数に対応する出力
        self.qnet_target = QNet(self.board_size * self.board_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        # 状態を1次元のベクトルに変換
        state = np.array(state)
        state = state[np.newaxis, :]  # (1, 19, 19)に変換
        state_flat = state.reshape(state.shape[0], -1)  # (1, 361)に変換

        # Qネットワークを使用してQ値を計算
        qs = self.qnet(state_flat)  # Q値の計算

        # Q値をNumPy配列に変換
        qs = qs.data  # VariableからNumPy配列に変換

        # 空いている場所を取得
        empty_positions = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if state[0][i][j] == 0]
        print(state)
        print(empty_positions)
        if 0 < self.epsilon:
            # ランダムに行動を選ぶ（探索）
            action = empty_positions[np.random.randint(0, len(empty_positions))]  # 空いている位置からランダムに選択
        else:
            # Q値が最大となる行動を選択（活用）
            action = empty_positions[np.argmax(qs[0])]
            
            
        return action


    

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        # Q値を計算
        qs = self.qnet(state)
        
        if action.shape[1] == 2:
            action = np.argmax(action, axis=1)
        
        q = qs[np.arange(self.batch_size), action]
        
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.qnet.save_weights(os.path.join(path, 'qnet.npz'))
        print(f"Model saved at {path}")

    def load(self, path):
        self.qnet.load_weights(os.path.join(path, 'qnet.npz'))
        print(f"Model loaded from {path}")


# === Main ===
episodes = 300
sync_interval = 20
## TODO: RandomGomoku.makeEnv() is not defined
env = GomokuEnv.GomokuEnv()
dqn_agent = DQNAgent()
random_agent = RandomAgent()
reward_history = []

# 学習の実行
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:

        if env.current_player == 1:
            action = dqn_agent.get_action(state)  # DQNエージェントのターン
        else:
            action = random_agent.get_action(state)  # ランダムエージェントのターン
        
        
        next_state, reward, done, info = env.step(action)

        dqn_agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        dqn_agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 1 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))

# モデルの保存
save_path = './dqn_model'
dqn_agent.save(save_path)

# === Plot ===
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()

# === 学習済みモデルでCartPoleをプレイ ===
dqn_agent.load(save_path)  # モデルをロード
dqn_agent.epsilon = 0  # greedy policy
state = env.reset()
done = False
total_reward = 0

while not done:
    action = dqn_agent.get_action(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    total_reward += reward
    # env.render()
print('Total Reward:', total_reward)
