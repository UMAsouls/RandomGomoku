"""
デュアルネットワークモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import device, NETWORK_NUM_CHANNELS, NETWORK_NUM_RES_BLOCKS, NETWORK_DROPOUT_RATE


class DualNetwork(nn.Module):
    """方策と価値を出力するニューラルネットワーク"""
    
    def __init__(self, board_size, num_channels=NETWORK_NUM_CHANNELS, num_res_blocks=NETWORK_NUM_RES_BLOCKS):
        super(DualNetwork, self).__init__()
        self.board_size = board_size
        
        # 入力層 (4チャネル入力)
        self.conv_input = nn.Conv2d(4, num_channels, 3, stride=1, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 階層的な畳み込み層 (32→64→128)
        self.conv1 = nn.Conv2d(num_channels, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 残差ブロック (128チャネルで構築)
        self.res_blocks = nn.ModuleList([
            self._build_res_block(128) for _ in range(num_res_blocks)
        ])
        
        # 方策ヘッド
        self.policy_conv = nn.Conv2d(128, 4, 1, stride=1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)
        
        # 価値ヘッド (勝率回帰用：0から1の範囲)
        self.value_conv = nn.Conv2d(128, 2, 1, stride=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.value_dropout = nn.Dropout(NETWORK_DROPOUT_RATE)
        self.value_fc2 = nn.Linear(64, 1)
        
        # すべてのパラメータを0で初期化
        for p in self.parameters():
            p.data.zero_()
    
    def _build_res_block(self, num_channels):
        """残差ブロックを構築"""
        return nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels)
        )
    
    def forward(self, x, last_move=None, current_player=1):
        """
        ニューラルネットワークの順伝播
        
        Args:
            x: 盤面の状態 [batch, board_size, board_size] または [batch, channels, board_size, board_size]
            last_move: 相手の最後の手 (x, y) または None
            current_player: 現在のプレイヤー (1: 黒, -1: 白)
        """
        # 入力: [batch, board_size, board_size] -> [batch, 4, board_size, board_size]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        if x.size(1) == 1:
            # 1チャネルを4チャネルの特徴量に変換
            board_state = x.squeeze(1)  # [batch, board_size, board_size]
            batch_size = board_state.shape[0]
            
            # チャネル1: 現在のプレイヤーの石 (1)、それ以外 (0)
            player_stones = torch.zeros_like(board_state)
            player_stones[board_state == current_player] = 1.0
            
            # チャネル2: 相手プレイヤーの石 (1)、それ以外 (0)
            opponent_stones = torch.zeros_like(board_state)
            opponent_stones[board_state == -current_player] = 1.0
            
            # チャネル3: 相手の最後の手 (1)、それ以外 (0)
            last_move_plane = torch.zeros_like(board_state)
            if last_move is not None:
                # バッチ内の各盤面に最後の手を記録
                for i in range(batch_size):
                    if isinstance(last_move, tuple):
                        x_pos, y_pos = last_move
                        last_move_plane[i, y_pos, x_pos] = 1.0
                    elif torch.is_tensor(last_move) and last_move.shape[0] == batch_size:
                        x_pos, y_pos = last_move[i] % self.board_size, last_move[i] // self.board_size
                        last_move_plane[i, y_pos, x_pos] = 1.0
            
            # チャネル4: 現在のプレイヤーが黒なら全て1、白なら全て0
            player_color = torch.full_like(board_state, float(current_player == 1))
            
            # 4チャネルを結合
            x = torch.stack([player_stones, opponent_stones, last_move_plane, player_color], dim=1)
        
        # 入力層
        x = self.relu(self.bn_input(self.conv_input(x)))
        
        # 階層的な畳み込み層
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # 残差ブロック処理
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x += residual
            x = self.relu(x)
        
        # 方策ヘッド
        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 4 * self.board_size * self.board_size)
        policy_logits = self.policy_fc(policy)
        
        # 価値ヘッド
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 2 * self.board_size * self.board_size)
        value = self.relu(self.value_fc1(value))
        value = self.value_dropout(value)
        value_output = torch.sigmoid(self.value_fc2(value))
        
        return policy_logits, value_output
