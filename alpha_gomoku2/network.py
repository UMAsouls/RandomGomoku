import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from GomokuEnv import GomokuEnv
from RandomGomoku.Board import Board

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
        self.use_gpu = use_gpu
        self.board_size = board_size
        self.l2_const = 1e-4  # L2ペナルティの係数
        self.env = env
        # ポリシー・バリューネットワークモジュール
        if self.use_gpu:
            self.policy_value_net = Net(board_size, board_size).cuda()
        else:
            self.policy_value_net = Net(board_size, board_size)
        # オプティマイザ
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            self.load_model(model_file)
    def get_legal_positions(self, board: Board):
        #形式: 整数のリスト（例：[0, 1, 2, 5, 7, 10, ...]）
        # 意味: 各整数は盤面上の空いているマス目の位置を1次元のインデックスで表現
        # 範囲: 0 ～ (board_size × board_size - 1)
        state = board.GetBoardInt()
        legal_positions = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if state[y][x] == 0:
                    legal_positions.append(x+ y * self.board_size)
        # print(f"有効な手の数: {len(legal_positions)}")
        # print(f"有効な手の位置: {legal_positions}")
        return legal_positions
    
    def _board_to_state_input(self, board: Board):
        """
        盤面の状態をネットワークの入力形式に変換します。
        入力: Boardオブジェクト
        出力: 4x(盤面サイズ)x(盤面サイズ) のnumpy配列
        """
        board_state = board.GetBoardInt()
        current_player = self.env.current_player
        last_move = self.env.lastmove

        # 4つの特徴平面を準備
        # 0: 現在のプレイヤーの石
        # 1: 相手プレイヤーの石
        # 2: 最後の着手
        # 3: 手番の色
        square_state = np.zeros((4, self.board_size, self.board_size))

        # 0: 現在のプレイヤーの石, 1: 相手プレイヤーの石
        square_state[0] = (board_state == current_player)
        square_state[1] = (board_state == (3 - current_player)) # 相手プレイヤー (1 -> 2, 2 -> 1)

        # 2: 最後の着手
        if last_move is not None:
            y, x = last_move
            print(f"最後の着手: ({x}, {y})")
            square_state[2, y, x] = 1.0
        
        # 3: 手番の色 (黒番なら全面1.0)
        if current_player == 1: # 黒番
            square_state[3] = 1.0
        
        return square_state

    def policy_value(self, state_batch):
        """
        入力: 状態のバッチ
        出力: 行動確率と状態価値のバッチ
        """
        # リストをNumPy配列に変換
        if isinstance(state_batch, list):
            state_batch = np.array(state_batch)
        
        # 状態バッチを4次元形状に変換
        if len(state_batch.shape) == 2:
            # バッチサイズを推定（状態バッチの行数）
            batch_size = state_batch.shape[0]
            # 想定される形状: (batch_size, 4, board_size, board_size)
            expected_size = 4 * self.board_size * self.board_size
            if state_batch.shape[1] == expected_size:
                state_batch = state_batch.reshape(batch_size, 4, self.board_size, self.board_size)
        
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()
    def policy_value_fn(self,board:Board):
        """
        入力: 盤面の状態
        出力: (行動, 確率) のタプルのリストと盤面の評価値
        """
        # stateから0の部分だけを抽出
        legal_positions = self.get_legal_positions(board)
        current_state = self._board_to_state_input(board)
        current_state = np.ascontiguousarray(current_state.reshape(
                -1, 4, self.board_size, self.board_size))
        
        if self.use_gpu:
                log_act_probs, value = self.policy_value_net(
                        Variable(torch.from_numpy(current_state)).cuda().float())
                act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
                value = value.data.cpu().numpy()[0][0]
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            value = value.data.numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """学習を1ステップ実行します"""
        # リストをNumPy配列に変換
        if isinstance(mcts_probs, list):
            mcts_probs = np.array(mcts_probs)
        if isinstance(winner_batch, list):
            winner_batch = np.array(winner_batch)
            
        # Variableにラップ
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # パラメータの勾配をゼロに設定
        self.optimizer.zero_grad()
        # 学習率を設定
        set_learning_rate(self.optimizer, lr)

        # 順伝播
        log_act_probs, value = self.policy_value_net(state_batch)
        # 損失関数を定義: loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # L2ペナルティはオプティマイザに組み込まれていることに注意
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # 逆伝播と最適化
        loss.backward()
        self.optimizer.step()
        # ポリシーエントロピーを計算（モニタリング用）
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()
        
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
        if self.use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'
        net_params = torch.load(model_file, map_location=device)
        self.policy_value_net.load_state_dict(net_params)
        print(f"モデルをロードしました: {model_file}")
        print(f"モデルをロードしました: {model_file}")
