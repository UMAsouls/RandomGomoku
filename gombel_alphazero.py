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
import datetime

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DualNetwork(nn.Module):
    """方策と価値を出力するニューラルネットワーク"""
    def __init__(self, board_size, num_channels=256):
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
        # Gumbel探索のための保存値
        self.action_scores = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class GombelMCTS:
    """Gombel分布を用いたモンテカルロ木探索の実装"""
    def __init__(self, model, num_simulations=800, c_puct=1.0, alpha=0.03):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.board_size = model.board_size
        self.alpha = alpha  # Gumbel分布のノイズスケール

    def _ucb_score(self, parent, child, gumbel_noise=None):
        """UCB (Upper Confidence Bound) スコアの計算 + Gumbelノイズ"""
        prior_score = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        if child.visit_count > 0:
            value_score = -child.value()
        else:
            value_score = 0
        
        # Gumbelノイズを追加（探索時のみ）
        if gumbel_noise is not None:
            return value_score + prior_score + gumbel_noise
        return value_score + prior_score
    
    def _sample_gumbel(self, shape, eps=1e-10):
        """Gumbel(0,1)分布からサンプリング"""
        u = np.random.random(shape)
        return -np.log(-np.log(u + eps) + eps)
    
    def search(self, state, temperature=1.0, add_noise=True):
        """与えられた状態に基づいてGombel MCTSを実行"""
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
        for move in legal_moves:
            policy_legal[move] = policy[move]
            
        # 合法手がある場合は正規化
        if len(legal_moves) > 0:
            policy_legal = policy_legal / np.sum(policy_legal)
            
        # Dirichlet ノイズを追加してルートの探索を促進
        if add_noise and len(legal_moves) > 0:
            # 修正：Dirichletノイズを同じ形状に合わせる
            noise = np.zeros(self.board_size * self.board_size)
            dirichlet_noise = np.random.dirichlet([0.3] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                noise[move] = dirichlet_noise[i]
            policy_legal = 0.75 * policy_legal + 0.25 * noise
            
        # 子ノードの初期化
        for move in legal_moves:
            root.children[move] = MCTSNode(policy_legal[move])
        
        # アクションスコアの初期化
        root.action_scores = np.zeros(self.board_size * self.board_size)
        
        # シミュレーション実行
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()
            
            # 葉ノードを見つける
            while node.expanded():
                # Gumbel-Max Sampling による探索
                actions = list(node.children.keys())
                if len(actions) == 0:
                    break
                
                if node == root and add_noise:
                    # Gumbelノイズをルートノードの子に追加
                    gumbel_noises = self.alpha * self._sample_gumbel(len(actions))
                    scores = [
                        self._ucb_score(node, node.children[a], gumbel_noise=gumbel_noises[i])
                        for i, a in enumerate(actions)
                    ]
                else:
                    scores = [self._ucb_score(node, node.children[a]) for a in actions]
                
                best_idx = np.argmax(scores)
                action = actions[best_idx]
                
                # 行動実行
                x, y = action % self.board_size, action // self.board_size
                current_state[y][x] = -1 if np.sum(current_state == 1) > np.sum(current_state == -1) else 1
                
                node = node.children[action]
                search_path.append(node)
            
            # 葉ノードの状態を評価
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
        for action, child in root.children.items():
            root.action_scores[action] = child.visit_count
        
        # 温度パラメータを使って方策を計算
        if temperature == 0:
            # グリーディー選択
            action = np.argmax(root.action_scores)
            mcts_policy = np.zeros(self.board_size * self.board_size)
            mcts_policy[action] = 1.0
        else:
            # 温度付き方策
            visit_count_distribution = root.action_scores ** (1 / temperature)
            if np.sum(visit_count_distribution) > 0:
                mcts_policy = visit_count_distribution / np.sum(visit_count_distribution)
            else:
                mcts_policy = np.ones(self.board_size * self.board_size) / (self.board_size * self.board_size)
        
        return mcts_policy, root.action_scores
    
    def _get_legal_moves(self, state):
        """合法手のリストを返す"""
        legal_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if state[y][x] == 0:  # 空のセル
                    legal_moves.append(y * self.board_size + x)
        return legal_moves

    def _is_terminal(self, state):
        """終端状態かどうかを返す"""
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
        """勝者に基づく価値を返す"""
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
    
    # MCTSの初期化
    mcts = GombelMCTS(model, num_simulations=100)  # 訓練時は計算量削減のため100回に設定
    
    # 環境の初期化
    env = GomokuEnv(board_size=board_size)
    
    game_memory = []
    state = env.board.GetBoardInt()
    
    done = False
    current_player = 1
    
    # ゲームの進行に応じて温度を変化させる
    temperature_schedule = lambda move: 1.0 if move < 10 else 0.5 if move < 20 else 0.2
    
    # ゲーム実行
    move_count = 0
    while not done:
        # 現在の温度を取得
        temperature = temperature_schedule(move_count)
        
        # MCTSで方策を計算（Gombel分布を利用）
        mcts_policy, action_scores = mcts.search(np.array(state), temperature=temperature)
        
        # 訓練データの保存
        game_memory.append((np.array(state), mcts_policy, current_player))
        
        # 行動の選択（温度に基づく確率的選択）
        if np.random.random() < 0.03:  # わずかな確率でランダム探索
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
        move_count += 1
    
    # ゲーム終了後、報酬を割り当ててリプレイバッファに追加
    final_value = reward.item()
    for hist_state, hist_policy, hist_player in game_memory:
        # プレイヤーに応じた報酬の調整
        adjusted_value = final_value * hist_player
        replay_buffer.add(hist_state, hist_policy, adjusted_value)
    
    # 結果をキューに送信
    result_queue.put((game_idx, final_value))

def train_network(model, replay_buffer, epochs=1, batch_size=128, lr=0.001):
    """ネットワークの訓練"""
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

class GombelAlphaZero:
    """Gombel AlphaZeroの実装"""
    def __init__(self, board_size=19, num_iterations=100, num_self_play_games=100,
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
        self.replay_buffer = ReplayBuffer(capacity=500000)
        
        # モデルのチェックポイントパス
        self.model_path = os.path.join(checkpoint_dir, f'gombel_alphazero_{board_size}.pth')
        
        # 既存のモデルをロード
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path))
                print(f"モデルを読み込みました: {self.model_path}")
            except:
                print("新規モデルを初期化します")
        else:
            print("新規モデルを初期化します")
    
    def train(self):
        """Gombel AlphaZeroの訓練ループ"""
        for iteration in range(self.num_iterations):
            start_time = time.time()
            print(f"\nイテレーション {iteration+1}/{self.num_iterations}")
            
            # 1. 自己対戦でデータ生成（並列）
            print("自己対戦でデータを生成中...")
            self._generate_self_play_data()
            
            # 2. ニューラルネットワークの訓練
            print("ニューラルネットワークを訓練中...")
            policy_loss, value_loss = train_network(self.model, self.replay_buffer)
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            # 3. モデルの保存
            torch.save(self.model.state_dict(), self.model_path)
            print(f"モデルを保存しました: {self.model_path}")
            
            # 4. タイムスタンプ付きでモデルとログファイルを保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # タイムスタンプ付きのモデルファイル名
            timestamp_model_path = os.path.join(self.checkpoint_dir, f'gombel_alphazero_{self.board_size}_{timestamp}.pth')
            
            # タイムスタンプ付きでモデルを保存
            torch.save(self.model.state_dict(), timestamp_model_path)
            print(f"タイムスタンプ付きモデルを保存しました: {timestamp_model_path}")
            
            # トレーニング情報をログファイルに保存
            log_filename = os.path.join(self.log_dir, f'gombel_training_log_{timestamp}.txt')
            
            with open(log_filename, 'w') as f:
                f.write(f"Gombel Training Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Latest Model Path: {self.model_path}\n")
                f.write(f"Timestamped Model Path: {timestamp_model_path}\n")
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
    
    def _generate_self_play_data(self):
        """自己対戦データの生成（並列実行）"""
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
        """自己対戦プロセス（Gombel分布を使用）"""
        # モデルのロード
        model = DualNetwork(board_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # MCTSの初期化（Gombel版）
        mcts = GombelMCTS(model, num_simulations=100, alpha=0.03)
        
        for game_idx in game_indices:
            # 環境の初期化
            env = GomokuEnv(board_size=board_size)
            
            game_memory = []
            state = env.board.GetBoardInt()
            
            done = False
            current_player = 1
            move_count = 0
            
            # ゲーム実行
            while not done:
                state_array = np.array(state)
                
                # 温度の設定（ゲームの進行に応じて変化）
                temperature = 1.0 if move_count < 10 else 0.5 if move_count < 20 else 0.25
                
                # MCTSで方策を計算（Gombel分布を使用）
                mcts_policy, action_scores = mcts.search(
                    state_array, 
                    temperature=temperature,
                    add_noise=(move_count < 15)  # 序盤だけノイズを追加
                )
                
                # 訓練データの保存
                game_memory.append((state_array.copy(), mcts_policy, current_player))
                
                # 行動の選択
                if move_count < 5 or np.random.random() < 0.1:  # 最初の数ステップと小確率でより探索的に
                    action_idx = np.random.choice(len(mcts_policy), p=mcts_policy)
                else:
                    # より確定的な選択
                    actions = np.where(mcts_policy > 0)[0]
                    if len(actions) > 0:
                        action_idx = actions[np.argmax(mcts_policy[actions])]
                    else:
                        legal_moves = [(x, y) for y in range(board_size) for x in range(board_size) if state[y][x] == 0]
                        if not legal_moves:
                            break
                        x, y = random.choice(legal_moves)
                        action_idx = y * board_size + x
                
                action = (action_idx % board_size, action_idx // board_size)
                
                # 環境での行動実行
                next_state, reward, done, _ = env.step(action)
                state = next_state.cpu().numpy()
                current_player *= -1  # プレイヤー交代
                move_count += 1
            
            # ゲーム終了後、報酬を割り当て
            final_value = reward.item()
            
            for hist_state, hist_policy, hist_player in game_memory:
                # プレイヤーに応じた報酬の調整
                adjusted_value = final_value * hist_player
                shared_buffer.append((hist_state, hist_policy, adjusted_value))
            
            # 結果をキューに送信
            result_queue.put((game_idx, final_value))

    def play_against_human(self):
        """人間との対戦"""
        env = GomokuEnv(board_size=self.board_size)
        state = env.board.GetBoardInt()
        
        # Gombel MCTSの初期化
        mcts = GombelMCTS(self.model, num_simulations=800, alpha=0.01)  # 実戦時は探索を深くし、Gumbel温度は低く
        
        done = False
        human_first = input("先手で始めますか？ (y/n): ").lower() == 'y'
        
        if not human_first:
            # AIの手番
            mcts_policy, _ = mcts.search(np.array(state), temperature=0.1, add_noise=False)
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
                mcts_policy, _ = mcts.search(np.array(state), temperature=0.1, add_noise=False)
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
    
    # Gombel AlphaZeroの初期化
    gombel_az = GombelAlphaZero(
        board_size=board_size,
        num_iterations=1,
        num_self_play_games=10
    )
    
    # 訓練を実行
    gombel_az.train()
    
    # 人間との対戦
    # gombel_az.play_against_human()
