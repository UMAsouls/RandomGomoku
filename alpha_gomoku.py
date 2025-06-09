import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from collections import deque
from tqdm import tqdm
from GomokuEnv import GomokuEnv
import datetime  # 時間取得のためのモジュールを追加
import matplotlib.pyplot as plt  # グラフ作成用にmatplotlibをインポート

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DualNetwork(nn.Module):
    """方策と価値を出力するニューラルネットワーク"""
    def __init__(self, board_size, num_channels=512):  # チャネル数を増加
        super(DualNetwork, self).__init__()
        self.board_size = board_size
        
        # 共通の畳み込み層
        self.conv1 = nn.Conv2d(1, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        
        # バッチ正規化
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)
        
        # 方策ヘッド
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 価値ヘッド
        self.value_conv = nn.Conv2d(num_channels, 1, 1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # 入力: [batch, board_size, board_size] -> [batch, 1, board_size, board_size]
        x = x.view(-1, 1, self.board_size, self.board_size)
        
        # 共通の畳み込み層
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 方策ヘッド
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy_logits = policy.log_softmax(dim=1)
        
        # 価値ヘッド
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value

class MCTSNode:
    """モンテカルロ木探索のノード"""
    def __init__(self, prior=0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    """モンテカルロ木探索の実装"""
    def __init__(self, model, num_simulations=800, c_puct=1.0, use_gumbel=False, gumbel_scale=0.1):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.board_size = model.board_size
        self.use_gumbel = use_gumbel
        self.gumbel_scale = gumbel_scale

    def _ucb_score(self, parent, child, action):
        """UCB (Upper Confidence Bound) スコアの計算"""
        prior_score = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        if child.visit_count > 0:
            value_score = -child.value()
        else:
            value_score = 0
        
        # 勝ち確定手なら無限大のスコア
        if parent.state is not None and self._is_winning_move(parent.state, action):
            return float('inf')
        
        score = value_score + prior_score
        
        # Gumbelノイズを追加（有効な場合）
        if self.use_gumbel:
            # Gumbel(0, scale)分布からノイズを生成
            noise = np.random.gumbel(0, self.gumbel_scale)
            score += noise
                
        return score
    
    def _is_winning_move(self, state, move):
        """与えられた手が勝利に繋がるかチェックする軽量版"""
        x, y = move % self.board_size, move // self.board_size
        if state[y][x] != 0:  # すでに石が置かれている場合
            return False
        
        # プレイヤーを特定
        player = 1 if np.sum(state == 1) == np.sum(state == -1) else -1
        
        # 方向ベクトル定義: 横、縦、右下、左下
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # 自分自身
            
            # 正方向
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and state[ny][nx] == player:
                count += 1
                nx += dx
                ny += dy
            
            # 負方向
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and state[ny][nx] == player:
                count += 1
                nx -= dx
                ny -= dy
            
            if count >= 5:
                return True
                
        return False
    
    def search(self, state):
        """与えられた状態に基づいてMCTSを実行"""
        root = MCTSNode(0)
        root.state = state.copy()
        
        # 評価関数から方策と価値を取得
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        
        policy = torch.exp(policy_logits).squeeze(0).cpu().numpy()
        legal_moves = self._get_legal_moves(state)
        
        # 方策を合法手に制限
        policy_legal = np.zeros(self.board_size * self.board_size)
        
        # まず勝ち確定手をチェック
        winning_moves = []
        for move in legal_moves:
            if self._is_winning_move(state, move):
                winning_moves.append(move)
        
        # 勝ち確定手がある場合は、それらに極めて高い確率を設定
        if winning_moves:
            for move in winning_moves:
                policy_legal[move] = 1.0
            # 勝ち確定手のみに確率を集中させる
            policy_legal = policy_legal / np.sum(policy_legal)
        else:
            # 通常通り方策を使用
            for move in legal_moves:
                policy_legal[move] = policy[move]
                
            # 合法手がある場合は正規化
            if len(legal_moves) > 0:
                policy_legal = policy_legal / np.sum(policy_legal)
        
        # 子ノードの初期化
        for move in legal_moves:
            root.children[move] = MCTSNode(policy_legal[move])
        
        # シミュレーション実行
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()
            
            # 葉ノードを見つける
            while node.expanded():
                action, node = self._select_child(node)
                x, y = action % self.board_size, action // self.board_size
                current_state[y][x] = -1 if np.sum(current_state == 1) > np.sum(current_state == -1) else 1
                search_path.append(node)
            
            # 葉ノードの状態を評価
            # parent = search_path[-2]
            leaf_state = current_state
            
            # ターミナル状態かチェック
            is_terminal = self._is_terminal(leaf_state)
            value = 0
            
            if is_terminal:
                # ゲーム終了：勝者に基づいて価値を設定
                value = self._get_winner_value(leaf_state)
            else:
                # ノードを展開
                leaf_node = search_path[-1]
                leaf_node.state = leaf_state.copy()
                
                # ニューラルネットワークで評価
                leaf_tensor = torch.tensor(leaf_state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    policy_logits, value = self.model(leaf_tensor)
                
                policy = torch.exp(policy_logits).squeeze(0).cpu().numpy()
                value = value.item()
                
                # 合法手の取得と方策の正規化
                legal_moves = self._get_legal_moves(leaf_state)
                policy_legal = np.zeros(self.board_size * self.board_size)
                for move in legal_moves:
                    policy_legal[move] = policy[move]
                
                if len(legal_moves) > 0:
                    policy_legal = policy_legal / np.sum(policy_legal)
                
                # 子ノードの作成
                for move in legal_moves:
                    leaf_node.children[move] = MCTSNode(policy_legal[move])
            
            # バックプロパゲーション
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value = -value  # 交互に手番が変わるので、価値を反転
        
        # 訪問回数に基づく方策の計算
        visit_counts = np.zeros(self.board_size * self.board_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # 温度パラメータを使って方策を計算
        temperature = 1.0
        if temperature == 0:
            # グリーディー選択
            action = np.argmax(visit_counts)
            mcts_policy = np.zeros(self.board_size * self.board_size)
            mcts_policy[action] = 1.0
        else:
            # 温度付き方策
            visit_count_distribution = visit_counts ** (1 / temperature)
            if np.sum(visit_count_distribution) > 0:
                mcts_policy = visit_count_distribution / np.sum(visit_count_distribution)
            else:
                mcts_policy = np.ones(self.board_size * self.board_size) / (self.board_size * self.board_size)
        
        return mcts_policy
    
    def _select_child(self, node):
        """UCBスコアに基づいて子ノードを選択"""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            score = self._ucb_score(node, child, action)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def _get_legal_moves(self, state):
        """合法手のリストを返す（既に石が置かれている位置から2マス以内の空きマスのみ）"""
        legal_moves = set()
        stones_exist = False
        
        # 盤面上の全ての石を探索
        for y in range(self.board_size):
            for x in range(self.board_size):
                if state[y][x] != 0:  # 石が置かれている場合
                    stones_exist = True
                    # 周囲2マス以内の空きマスを合法手に追加
                    for dy in range(-2, 3):  # -2, -1, 0, 1, 2
                        for dx in range(-2, 3):  # -2, -1, 0, 1, 2
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and state[ny][nx] == 0:
                                legal_moves.add(ny * self.board_size + nx)
        
        
        return list(legal_moves)

    def _augment_data(self, state, policy):
        """盤面と方策の対称変換を行い、8つの等価なデータを生成"""
        # 元の状態と方策
        states, policies = [state], [policy]
        
        # 90度、180度、270度回転
        for i in range(1, 4):
            rot_state = np.rot90(state, k=i)
            rot_policy = self._rotate_policy(policy, i)
            states.append(rot_state)
            policies.append(rot_policy)
        
        # 水平反転とその回転
        flip_state = np.fliplr(state)
        flip_policy = self._flip_policy_horizontally(policy)
        states.append(flip_state)
        policies.append(flip_policy)
        
        # 水平反転した状態の90度、180度、270度回転
        for i in range(1, 4):
            rot_flip_state = np.rot90(flip_state, k=i)
            rot_flip_policy = self._rotate_policy(flip_policy, i)
            states.append(rot_flip_state)
            policies.append(rot_flip_policy)
            
        return states, policies

    def _rotate_policy(self, policy, k=1):
        """方策を90度×k回転させる"""
        policy_2d = policy.reshape(self.board_size, self.board_size)
        rotated_policy_2d = np.rot90(policy_2d, k=k)
        return rotated_policy_2d.flatten()
    
    def _flip_policy_horizontally(self, policy):
        """方策を水平方向に反転させる"""
        policy_2d = policy.reshape(self.board_size, self.board_size)
        flipped_policy_2d = np.fliplr(policy_2d)
        return flipped_policy_2d.flatten()

    def _is_terminal(self, state):
        """終端状態かどうかを返す（簡易実装）"""
        # 水平方向
        for y in range(self.board_size):
            for x in range(self.board_size - 4):
                if state[y][x] != 0 and all(state[y][x] == state[y][x+i] for i in range(5)):
                    return True
                    
        # 垂直方向
        for y in range(self.board_size - 4):
            for x in range(self.board_size):
                if state[y][x] != 0 and all(state[y+i][x] == state[y][x] for i in range(5)):
                    return True
                    
        # 右下がり対角線
        for y in range(self.board_size - 4):
            for x in range(self.board_size - 4):
                if state[y][x] != 0 and all(state[y+i][x+i] == state[y][x] for i in range(5)):
                    return True
                    
        # 左下がり対角線
        for y in range(self.board_size - 4):
            for x in range(4, self.board_size):
                if state[y][x] != 0 and all(state[y+i][x-i] == state[y][x] for i in range(5)):
                    return True
        
        # 引き分けまたは続行中
        return np.count_nonzero(state == 0) == 0
    
    def _get_winner_value(self, state):
        """勝者に基づく価値を返す（簡易実装）"""
        # 水平方向
        for y in range(self.board_size):
            for x in range(self.board_size - 4):
                if state[y][x] != 0 and all(state[y][x] == state[y][x+i] for i in range(5)):
                    return state[y][x]  # 勝者の値 (1 or -1)
                    
        # 垂直方向
        for y in range(self.board_size - 4):
            for x in range(self.board_size):
                if state[y][x] != 0 and all(state[y+i][x] == state[y][x] for i in range(5)):
                    return state[y][x]
                    
        # 右下がり対角線
        for y in range(self.board_size - 4):
            for x in range(self.board_size - 4):
                if state[y][x] != 0 and all(state[y+i][x+i] == state[y][x] for i in range(5)):
                    return state[y][x]
                    
        # 左下がり対角線
        for y in range(self.board_size - 4):
            for x in range(4, self.board_size):
                if state[y][x] != 0 and all(state[y+i][x-i] == state[y][x] for i in range(5)):
                    return state[y][x]
        
        # 引き分け
        return 0

class ReplayBuffer:
    """経験リプレイバッファ"""
    def __init__(self, capacity=100000):
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

class SelfPlayDataset(Dataset):
    """自己対戦データセット"""
    def __init__(self, states, policies, values):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.policies = torch.tensor(policies, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def self_play_worker(model_path, board_size, replay_buffer, game_idx, result_queue):
    """自己対戦を行い、訓練データを生成するワーカー関数"""
    # モデルの読み込み
    model = DualNetwork(board_size)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        pass
    model.to(device)
    model.eval()
    
    # MCTSの初期化 (Gumbelを使用)
    mcts = MCTS(model, num_simulations=400, use_gumbel=True, gumbel_scale=0.05)  # 訓練時も計算量を増加
    
    # 環境の初期化
    env = GomokuEnv(board_size=board_size)
    
    game_memory = []
    state = env.board.GetBoardInt()
    
    done = False
    current_player = 1
    
    # ゲーム実行
    while not done:
        # MCTSで方策を計算
        mcts_policy = mcts.search(np.array(state))
        
        # 訓練データの保存
        game_memory.append((np.array(state), mcts_policy, current_player))
        
        # 行動の選択
        if np.random.random() < 0.05:  # 探索のための確率的選択
            legal_moves = [(x, y) for y in range(board_size) for x in range(board_size) if state[y][x] == 0]
            if legal_moves:
                action = random.choice(legal_moves)
            else:
                break
        else:
            # 方策から行動を選択
            action_idx = np.random.choice(len(mcts_policy), p=mcts_policy)
            action = (action_idx % board_size, action_idx // board_size)
        
        # 環境での行動実行
        state, reward, done, _ = env.step(action)
        state = state.cpu().numpy()
        current_player *= -1  # プレイヤー交代
    
    # ゲーム終了後、報酬を割り当ててリプレイバッファに追加
    final_value = reward.item()
    for hist_state, hist_policy, hist_player in game_memory:
        # プレイヤーに応じた報酬の調整
        adjusted_value = final_value * hist_player
        replay_buffer.add(hist_state, hist_policy, adjusted_value)
    
    # 結果をキューに送信
    result_queue.put((game_idx, final_value))

def train_network(model, replay_buffer, epochs=10, batch_size=256, lr=0.0005):
    """ニューラルネットワークの訓練 - パラメータ調整"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # リプレイバッファからデータを取得
    if len(replay_buffer) < batch_size:
        return 0, 0  # データが十分でない場合はスキップ
        
    states, policies, values = replay_buffer.sample(min(len(replay_buffer), 10000))
    
    # データセットとデータローダーの作成
    dataset = SelfPlayDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_policy_loss = 0
    total_value_loss = 0
    
    for _ in range(epochs):
        for batch_states, batch_policies, batch_values in dataloader:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device)
            
            # 予測
            policy_logits, value_preds = model(batch_states)
            
            # 損失の計算
            policy_loss = -torch.sum(batch_policies * policy_logits) / batch_states.size(0)
            value_loss = F.mse_loss(value_preds.view(-1), batch_values)
            
            # 正則化項（L2正則化はoptimizer内で行う）
            loss = policy_loss + value_loss
            
            # 勾配の計算と更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    
    avg_policy_loss = total_policy_loss / (len(dataloader) * epochs)
    avg_value_loss = total_value_loss / (len(dataloader) * epochs)
    
    return avg_policy_loss, avg_value_loss

class AlphaZero:
    """AlphaZeroの実装"""
    def __init__(self, board_size=19, num_iterations=100, num_self_play_games=64,
                 checkpoint_dir='models', log_dir='logs'):
        self.board_size = board_size
        self.num_iterations = num_iterations
        self.num_self_play_games = num_self_play_games
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # ディレクトリの作成
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # モデルの初期化
        self.model = DualNetwork(board_size).to(device)
        
        # リプレイバッファの初期化
        self.replay_buffer = ReplayBuffer(capacity=17000)
        
        # タイムスタンプの作成（モデルとログの両方で使用）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # モデルのベース名を設定
        self.model_base_name = f'alpha_gomoku_{board_size}'
        
        # モデルのチェックポイントパス
        self.model_path = os.path.join(checkpoint_dir, f'{self.model_base_name}_{timestamp}.pth')
        
        # 時間ベースのログディレクトリを作成
        self.timestamp_log_dir = os.path.join(log_dir, f'log_{timestamp}')
        os.makedirs(self.timestamp_log_dir, exist_ok=True)
        
        # 損失履歴を記録するリストを追加
        self.policy_loss_history = []
        self.value_loss_history = []
        # イテレーション番号を追跡するリストを追加
        self.iterations = []
        
        # 既存のモデルをロード
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path))
                print(f"モデルを読み込みました: {self.model_path}")
            except:
                print("新規モデルを初期化します")
        else:
            print("新規モデルを初期化します")
            # 初期モデルの保存
            self.save_model("initial")
    
    def save_model(self, stage, iteration=None):
        """モデルを保存するヘルパーメソッド"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ファイル名の生成 
        if iteration is not None:
            model_filename = f'{self.model_base_name}_iter{iteration}_{stage}_{timestamp}.pth'
        else:
            model_filename = f'{self.model_base_name}_{stage}_{timestamp}.pth'
            
        save_path = os.path.join(self.checkpoint_dir, model_filename)
        
        # モデルの保存
        torch.save(self.model.state_dict(), save_path)
        print(f"モデルを保存しました: {save_path}")
        
        # 最新のモデルパスを更新
        self.model_path = save_path
        return save_path
    
    def train(self):
        """AlphaZeroの訓練ループ"""
        for iteration in range(self.num_iterations):
            start_time = time.time()
            print(f"\nイテレーション {iteration+1}/{self.num_iterations}")
            
            # 1. 自己対戦でデータ生成（並列）
            print("自己対戦でデータを生成中...")
            self._generate_self_play_data()
            
            # 自己対戦後のモデルを保存
            self.save_model("after_selfplay", iteration+1)
            
            # 自己対戦後の状態をグラフ化して保存
            self._plot_loss_history(stage="after_selfplay", iteration=iteration+1)
            
            # 2. ニューラルネットワークの訓練
            print("ニューラルネットワークを訓練中...")
            policy_loss, value_loss = train_network(self.model, self.replay_buffer)
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            # 損失履歴に追加
            self.policy_loss_history.append(policy_loss)
            self.value_loss_history.append(value_loss)
            self.iterations.append(iteration+1)
            
            # 訓練後のモデルを保存
            self.save_model("trained", iteration+1)
            
            # 訓練後の状態をグラフ化して保存
            self._plot_loss_history(stage="after_training", iteration=iteration+1)
            
            # 3. トレーニング情報をログファイルに保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(self.timestamp_log_dir, f'training_log_{iteration+1}_{timestamp}.txt')
            
            with open(log_filename, 'w') as f:
                f.write(f"Training Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Path: {self.model_path}\n")
                f.write(f"Iteration: {iteration+1}/{self.num_iterations}\n")
                f.write(f"Policy Loss: {policy_loss:.6f}\n")
                f.write(f"Value Loss: {value_loss:.6f}\n")
                f.write(f"Replay Buffer Size: {len(self.replay_buffer)}\n")
                f.write(f"Self-Play Games: {self.num_self_play_games}\n")
                f.write(f"Board Size: {self.board_size}\n")
                f.write(f"Training Duration: {time.time() - start_time:.2f} seconds\n")
            
            print(f"トレーニング情報をログに保存しました: {log_filename}")
            
            iteration_time = time.time() - start_time
            print(f"イテレーション完了: {iteration_time:.2f} 秒")
        
        # トレーニング終了時に最終的な損失グラフを保存
        self._plot_loss_history(final=True)
        
        # トレーニング完了時の最終モデルを保存
        self.save_model("final")
    
    def _plot_loss_history(self, final=False, stage=None, iteration=None):
        """損失の履歴をグラフ化して保存"""
        if len(self.policy_loss_history) == 0:
            return  # 損失データがない場合は何もしない
            
        plt.figure(figsize=(15, 10))
        
        # グラフタイトルに情報を追加
        title_suffix = ""
        if final:
            title_suffix = " (Final)"
        elif stage and iteration:
            title_suffix = f" ({stage}, Iteration {iteration})"
        
        # 2つのグラフを並べて表示
        plt.subplot(2, 1, 1)
        plt.plot(self.iterations, self.policy_loss_history, 'b-', marker='o')
        plt.title(f'Policy Loss History{title_suffix}')
        plt.xlabel('Iteration')
        plt.ylabel('Policy Loss')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.iterations, self.value_loss_history, 'r-', marker='o')
        plt.title(f'Value Loss History{title_suffix}')
        plt.xlabel('Iteration')
        plt.ylabel('Value Loss')
        plt.grid(True)
        
        # グラフ下部に現在の訓練状況を表示
        plt.figtext(0.5, 0.01, 
                   f"Board Size: {self.board_size}, Buffer Size: {len(self.replay_buffer)}, Games/Iter: {self.num_self_play_games}", 
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # ファイル名の作成
        parts = []
        if final:
            parts.append("final")
        if stage:
            parts.append(stage)
        if iteration:
            parts.append(f"iter{iteration}")
            
        prefix = "_".join(parts)
        if prefix:
            prefix += "_"
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.timestamp_log_dir, f'{prefix}loss_history_{timestamp}.png')
        
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        
        print(f"損失のグラフを保存しました: {plot_filename}")

    def _generate_self_play_data(self):
        """自己対戦データの生成（並列実行）"""
        # 現在の最新のモデルを使用
        torch.save(self.model.state_dict(), self.model_path)
        
        # マルチプロセッシングの準備
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        processes = []
        manager = mp.Manager()
        shared_buffer = manager.list()
        
        # プロセス数はCPUコア数かゲーム数の小さい方
        num_processes = min(mp.cpu_count(), self.num_self_play_games)
        games_per_process = self.num_self_play_games // num_processes
        
        print(f"並列プロセス数: {num_processes}, プロセスあたりのゲーム数: {games_per_process}")
        
        # プロセス作成と開始
        for i in range(num_processes):
            start_idx = i * games_per_process
            end_idx = (i + 1) * games_per_process if i < num_processes - 1 else self.num_self_play_games
            
            p = ctx.Process(
                target=self._self_play_process,
                args=(self.model_path, self.board_size, shared_buffer, range(start_idx, end_idx), result_queue)
            )
            p.start()
            processes.append(p)
        
        # 進捗表示
        pbar = tqdm(total=self.num_self_play_games, desc="自己対戦")
        completed_games = 0
        
        while completed_games < self.num_self_play_games:
            if not result_queue.empty():
                game_idx, result = result_queue.get()
                completed_games += 1
                pbar.update(1)
        
        pbar.close()
        
        # 全プロセス終了待ち
        for p in processes:
            p.join()
        
        # リプレイバッファにデータを追加
        for item in shared_buffer:
            state, policy, value = item
            self.replay_buffer.add(state, policy, value)
        
        print(f"リプレイバッファサイズ: {len(self.replay_buffer)}")

    def _self_play_process(self, model_path, board_size, shared_buffer, game_indices, result_queue):
        """自己対戦プロセス"""
        # モデルのロード
        model = DualNetwork(board_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # MCTSの初期化 (Gumbelを使用)
        mcts = MCTS(model, num_simulations=400, use_gumbel=True, gumbel_scale=0.05)  # シミュレーション回数増加
        
        for game_idx in game_indices:
            # 環境の初期化
            env = GomokuEnv(board_size=board_size)
            
            game_memory = []
            state = env.board.GetBoardInt()
            
            done = False
            current_player = 1
            
            # ゲーム実行
            while not done:
                state_array = np.array(state)
                # MCTSで方策を計算
                mcts_policy = mcts.search(state_array)
                
                # 訓練データの保存
                game_memory.append((state_array.copy(), mcts_policy, current_player))
                
                # 行動の選択（より探索的な選択）
                if len(game_memory) < 10 or np.random.random() < 0.1:  # 最初の数ステップは探索
                    action_idx = np.random.choice(len(mcts_policy), p=mcts_policy)
                else:
                    # より確定的な選択
                    action_idx = np.argmax(mcts_policy)
                
                action = (action_idx % board_size, action_idx // board_size)
                
                # 環境での行動実行
                next_state, reward, done, _ = env.step(action)
                state = next_state.cpu().numpy()
                current_player *= -1  # プレイヤー交代
            
            # ゲーム終了後、報酬を割り当て
            final_value = reward.item()
            
            for hist_state, hist_policy, hist_player in game_memory:
                # プレイヤーに応じた報酬の調整
                adjusted_value = final_value * hist_player
                
                # 対称性を活用してデータを拡張（8倍に）
                augmented_states, augmented_policies = mcts._augment_data(hist_state, hist_policy)
                for aug_state, aug_policy in zip(augmented_states, augmented_policies):
                    shared_buffer.append((aug_state, aug_policy, adjusted_value))
            
            # 結果をキューに送信
            result_queue.put((game_idx, final_value))

    def play_against_human(self):
        """人間との対戦"""
        env = GomokuEnv(board_size=self.board_size)
        state = env.board.GetBoardInt()
        
        # MCTSの初期化 (対戦時はGumbelを無効化)
        mcts = MCTS(self.model, num_simulations=800, use_gumbel=False)  # 実戦時はシミュレーション回数を増やす
        
        done = False
        human_first = input("先手で始めますか？ (y/n): ").lower() == 'y'
        
        if not human_first:
            # AIの手番
            mcts_policy = mcts.search(np.array(state))
            action_idx = np.argmax(mcts_policy)
            action = (action_idx % self.board_size, action_idx // self.board_size)
            state, reward, done, _ = env.step(action)
            state = state.cpu().numpy()
            env.render()
        
        while not done:
            # 人間の手番
            try:
                x = int(input(f"列 (0-{self.board_size-1}): "))
                y = int(input(f"行 (0-{self.board_size-1}): "))
                if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size or state[y][x] != 0:
                    print("無効な手です。再入力してください。")
                    continue
                action = (x, y)
                state, reward, done, _ = env.step(action)
                state = state.cpu().numpy()
                env.render()
                
                if done:
                    print("あなたの勝ちです！")
                    break
                
                # AIの手番
                mcts_policy = mcts.search(np.array(state))
                action_idx = np.argmax(mcts_policy)
                action = (action_idx % self.board_size, action_idx // self.board_size)
                state, reward, done, _ = env.step(action)
                state = state.cpu().numpy()
                env.render()
                
                if done:
                    print("AIの勝ちです！")
                
            except ValueError:
                print("数値を入力してください")
                continue

if __name__ == "__main__":
    # ボードサイズ（15x15は標準的な五目並べのサイズ）
    board_size = 7
    
    # AlphaZeroの初期化 - 本番用パラメータ
    alpha_zero = AlphaZero(
        board_size=board_size,
        num_iterations=100,     # 5120から100に削減 - 実用的なトレーニング回数
        num_self_play_games=64, # 256から64に削減 - 効率的なデータ生成数
    )
    
    # MCTSクラス内のnum_simulationsを調整
    # MCTS初期化時に num_simulations=800 を使用（自己対戦時は400程度）
    
    # 訓練を実行
    alpha_zero.train()
    
    # 人間との対戦
    # alpha_zero.play_against_human()
