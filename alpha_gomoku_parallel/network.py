import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from GomokuEnv import GomokuEnv
from RandomGomoku.Board import Board
import threading
import concurrent.futures
from torch.utils.data import DataLoader, TensorDataset

def set_learning_rate(optimizer, lr):
    """指定された値に学習率を設定します"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    """ポリシー・バリューネットワークのモジュール"""
    def __init__(self, board_size, board_height):
        super(Net, self).__init__()

        self.board_size = board_size
        self.board_height = board_height
        # 共通層
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 行動ポリシー層
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_size*board_height,
                                 board_size*board_height)
        # 状態価値層
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_size*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # 共通層
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 行動ポリシー層
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_size*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # 状態価値層
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_size*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyValueNet():
    """ポリシー・バリューネットワーク"""
    def __init__(self, board_size,
                 model_file=None, use_gpu=False,env:GomokuEnv=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.board_size = board_size
        self.l2_const = 1e-4  # L2ペナルティの係数
        self.env = env
        
        # デバイス設定
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # CPUワーカーの数を設定
        self.num_cpu_workers = 4
        # CPU処理用のThreadPoolExecutor
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpu_workers)
        
        # ポリシー・バリューネットワークモジュール
        self.policy_value_net = Net(board_size, board_size).to(self.device)
        
        # オプティマイザ
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            self.load_model(model_file)
    
    def __del__(self):
        """デストラクタでスレッドプールを終了"""
        if hasattr(self, 'cpu_executor'):
            self.cpu_executor.shutdown(wait=True)
    
    def _preprocess_board_cpu(self, board: Board):
        """CPUでボードの前処理を行う"""
        board_state = board.GetBoardInt()
        current_player = self.env.current_player
        last_move = self.env.lastmove

        # 4つの特徴平面を準備
        square_state = np.zeros((4, self.board_size, self.board_size))

        # 0: 現在のプレイヤーの石, 1: 相手プレイヤーの石
        square_state[0] = (board_state == current_player)
        square_state[1] = (board_state == (3 - current_player))

        # 2: 最後の着手
        if last_move is not None:
            y, x = last_move
            square_state[2, y, x] = 1.0
        
        # 3: 手番の色
        if current_player == 1:
            square_state[3] = 1.0
        
        return square_state
    
    def _get_legal_positions_cpu(self, board: Board):
        """CPUで合法手を取得"""
        state = board.GetBoardInt()
        legal_positions = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if state[y][x] == 0:
                    legal_positions.append(x+ y * self.board_size)
        return legal_positions

    def get_legal_positions(self, board: Board):
        """合法手を取得（CPUで実行）"""
        return self._get_legal_positions_cpu(board)
    
    def _board_to_state_input(self, board: Board):
        """CPUで盤面の状態をネットワークの入力形式に変換"""
        return self._preprocess_board_cpu(board)

    def policy_value(self, state_batch):
        """
        入力: 状態のバッチ
        出力: 行動確率と状態価値のバッチ
        """
        # CPUでデータ前処理
        if isinstance(state_batch, list):
            state_batch = np.array(state_batch)
        
        # 状態バッチを4次元形状に変換（CPUで実行）
        if len(state_batch.shape) == 2:
            batch_size = state_batch.shape[0]
            expected_size = 4 * self.board_size * self.board_size
            if state_batch.shape[1] == expected_size:
                state_batch = state_batch.reshape(batch_size, 4, self.board_size, self.board_size)
        
        # DataLoaderを使用してバッチ処理を効率化
        dataset = TensorDataset(torch.FloatTensor(state_batch))
        dataloader = DataLoader(dataset, batch_size=min(64, len(state_batch)), shuffle=False)
        
        all_act_probs = []
        all_values = []
        
        # バッチごとに処理
        for batch_data in dataloader:
            batch_tensor = batch_data[0].to(self.device)
            
            # GPU推論
            with torch.no_grad():
                log_act_probs, value = self.policy_value_net(batch_tensor)
            
            # CPUで後処理
            act_probs = np.exp(log_act_probs.cpu().numpy())
            value_np = value.cpu().numpy()
            
            all_act_probs.append(act_probs)
            all_values.append(value_np)
        
        # 結果を結合
        final_act_probs = np.concatenate(all_act_probs, axis=0)
        final_values = np.concatenate(all_values, axis=0)
        
        return final_act_probs, final_values
    def policy_value_fn(self,board:Board):
        """
        入力: 盤面の状態
        出力: (行動, 確率) のタプルのリストと盤面の評価値
        """
        # CPUで前処理を並列実行
        future_legal = self.cpu_executor.submit(self._get_legal_positions_cpu, board)
        future_state = self.cpu_executor.submit(self._preprocess_board_cpu, board)
        
        # 並列処理の結果を取得
        legal_positions = future_legal.result()
        current_state = future_state.result()
        
        # GPU推論用にデータを準備
        current_state = np.ascontiguousarray(current_state.reshape(
                -1, 4, self.board_size, self.board_size))
        
        state_tensor = torch.from_numpy(current_state).float().to(self.device)
        
        # GPU推論
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
        
        # CPUで後処理
        act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
        value = value.cpu().numpy()[0][0]
        
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """学習を1ステップ実行します"""
        # NumPy配列をTensorに変換
        if isinstance(state_batch, np.ndarray):
            state_batch = torch.FloatTensor(state_batch)
        if isinstance(mcts_probs, np.ndarray):
            mcts_probs = torch.FloatTensor(mcts_probs)
        if isinstance(winner_batch, np.ndarray):
            winner_batch = torch.FloatTensor(winner_batch)
            
        # DataLoaderを使用してバッチ処理を効率化
        dataset = TensorDataset(state_batch, mcts_probs, winner_batch)
        dataloader = DataLoader(dataset, batch_size=min(128, len(state_batch)), shuffle=True)
        
        # パラメータの勾配をゼロに設定
        self.optimizer.zero_grad()
        # 学習率を設定
        set_learning_rate(self.optimizer, lr)
        
        total_loss = 0
        total_entropy = 0
        num_batches = 0
        
        for batch_states, batch_mcts_probs, batch_winners in dataloader:
            # GPU転送
            batch_states = batch_states.to(self.device)
            batch_mcts_probs = batch_mcts_probs.to(self.device)
            batch_winners = batch_winners.to(self.device)

            # 順伝播
            log_act_probs, value = self.policy_value_net(batch_states)
            # 損失関数を計算
            value_loss = F.mse_loss(value.view(-1), batch_winners)
            policy_loss = -torch.mean(torch.sum(batch_mcts_probs*log_act_probs, 1))
            loss = value_loss + policy_loss
            
            # 逆伝播
            loss.backward()
            
            # ポリシーエントロピーを計算
            entropy = -torch.mean(
                    torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                    )
            
            total_loss += loss.item()
            total_entropy += entropy.item()
            num_batches += 1
        
        # 最適化ステップ
        self.optimizer.step()
        
        return total_loss / num_batches, total_entropy / num_batches
        
    def get_policy_param(self):
        """ネットワークのパラメータを取得します"""
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """モデルのパラメータをファイルに保存します"""
        net_params = self.get_policy_param()  # モデルのパラメータを取得
        torch.save(net_params, model_file)
        
    def load_model(self, model_file):
        """モデルのパラメータをファイルから読み込みます"""
        net_params = torch.load(model_file, map_location=self.device)
        self.policy_value_net.load_state_dict(net_params)
        print(f"モデルをロードしました: {model_file}")
        print(f"デバイス: {self.device}")
