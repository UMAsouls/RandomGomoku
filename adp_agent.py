import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

# ADPネットワークの定義（既存のコードを維持）
class ADPNetwork(nn.Module):
    def __init__(self):
        super(ADPNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(100)
        self.fc1 = nn.Linear(100 * 15 * 15, 1000)
        self.fc2 = nn.Linear(1000, 15 * 15)

    def forward(self, x):
        x = x.view(-1, 1, 15, 15)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, 100 * 15 * 15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# DualNetwork用の互換性レイヤー（alpha_gomokuからインポートもできる）
try:
    from alpha_gomoku import DualNetwork
except ImportError:
    class DualNetwork(nn.Module):
        """代替の簡易DualNetwork定義（インポートできない場合のフォールバック）"""
        def __init__(self, board_size, num_channels=512):
            super(DualNetwork, self).__init__()
            self.board_size = board_size
            
            # 基本構造だけ定義
            self.conv1 = nn.Conv2d(1, num_channels, 3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(num_channels)
            
            # 方策ヘッド（簡易版）
            self.policy_fc = nn.Linear(num_channels * board_size * board_size, board_size * board_size)
            
            # 価値ヘッド（簡易版）
            self.value_fc2 = nn.Linear(num_channels, 1)
        
        def forward(self, x):
            # 簡易実装
            batch_size = x.size(0)
            policy = torch.zeros(batch_size, self.board_size * self.board_size).log()
            value = torch.zeros(batch_size, 1)
            return policy, value

class ADPAgent:
    """ADPAgent - AlphaDynamicPolicyの実装"""
    
    def __init__(self, model_path=None, epsilon=0.05, use_dual_network=False, board_size=15):
        """ADPAgentの初期化"""
        self.epsilon = epsilon
        self.model_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_dual_network = use_dual_network
        self.board_size = board_size
        
        # モデルのロード
        try:
            if use_dual_network:
                # DualNetworkモデルを使用
                self.model = DualNetwork(board_size=board_size).to(self.device)
                if model_path and os.path.exists(model_path):
                    print(f"DualNetworkモデルをロードしています: {model_path}")
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    self.model_loaded = True
            else:
                # 従来のADPNetworkを使用
                self.model = ADPNetwork().to(self.device)
                if model_path and os.path.exists(model_path):
                    print(f"ADPモデルをロードしています: {model_path}")
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    self.model_loaded = True
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            self.model_loaded = False
    
    def get_action(self, state, player_id):
        """行動選択ロジック"""
        # 確率εでランダム行動
        if random.random() < self.epsilon or not self.model_loaded:
            if not self.model_loaded:
                print("ADPランダムな行動をとります。")
            # 有効な手だけからランダム選択
            valid_actions = [(x, y) for y in range(len(state)) for x in range(len(state[0])) if state[y][x] == 0]
            if valid_actions:
                return random.choice(valid_actions)
            return (0, 0)  # 有効な手がない場合のフォールバック
        
        # モデルによる行動選択
        with torch.no_grad():
            # ボードサイズに応じてテンソルを整形
            board_tensor = self._prepare_input(state, player_id)
            
            if self.use_dual_network:
                # DualNetworkの場合は方策を使う
                policy_logits, _ = self.model(board_tensor)
                logits = policy_logits.squeeze(0)
                
                # 無効な手にはマスク（-inf）を適用
                mask = self._create_action_mask(state)
                masked_logits = logits + mask
                
                # 最も確率の高い手を選択
                action_idx = torch.argmax(masked_logits).item()
            else:
                # ADPNetworkの場合
                logits = self.model(board_tensor).squeeze(0)
                
                # 無効な手にはマスク（-inf）を適用
                mask = self._create_action_mask(state)
                masked_logits = logits + mask
                
                # 最も確率の高い手を選択
                action_idx = torch.argmax(masked_logits).item()
            
            # インデックスを(x, y)座標に変換
            x = action_idx % len(state[0])
            y = action_idx // len(state[0])
            
            return (x, y)

    def _prepare_input(self, state, player_id):
        """状態入力の前処理"""
        # プレイヤーIDに応じてボード状態を調整
        if player_id == 2:  # 白の場合、ボード表現を反転
            board = np.array([[-1 if cell == 1 else 1 if cell == 2 else 0 for cell in row] for row in state])
        else:  # 黒の場合
            board = np.array([[1 if cell == 1 else -1 if cell == 2 else 0 for cell in row] for row in state])
        
        # モデルがDualNetworkの場合はボードサイズに揃える
        if self.use_dual_network and (len(state) != self.board_size or len(state[0]) != self.board_size):
            # サイズを揃える必要がある場合の処理
            # 中央部分を切り出すか、パディングするなどの対応が必要
            # ここでは簡易的に7x7の場合の例を示す
            if self.board_size == 7:
                # 15x15から中央の7x7を切り出す
                center = len(state) // 2
                offset = self.board_size // 2
                board = board[center-offset:center+offset+1, center-offset:center+offset+1]
            else:
                # その他のサイズの場合はリサイズが必要
                pass
        
        # テンソルに変換
        tensor = torch.FloatTensor(board).to(self.device)
        return tensor.unsqueeze(0)  # バッチ次元を追加
        
    def _create_action_mask(self, state):
        """無効な手にマスク（-inf）を適用する関数"""
        if self.use_dual_network:
            # DualNetworkのボードサイズに合わせたマスク
            mask = torch.zeros(self.board_size * self.board_size).to(self.device)
            
            # ボードサイズが異なる場合の変換
            if len(state) != self.board_size:
                # 中央部分を使用する例（15x15 -> 7x7）
                if self.board_size == 7 and len(state) == 15:
                    center = len(state) // 2
                    offset = self.board_size // 2
                    for y in range(self.board_size):
                        for x in range(self.board_size):
                            board_y = center - offset + y
                            board_x = center - offset + x
                            if state[board_y][board_x] != 0:  # 既に石がある
                                mask[y * self.board_size + x] = float('-inf')
                else:
                    # その他のサイズ変換（必要に応じて実装）
                    pass
            else:
                # ボードサイズが一致している場合
                for y in range(self.board_size):
                    for x in range(self.board_size):
                        if state[y][x] != 0:  # 既に石がある
                            mask[y * self.board_size + x] = float('-inf')
        else:
            # 通常のADPモデル用（15x15を想定）
            mask = torch.zeros(15 * 15).to(self.device)
            for y in range(len(state)):
                for x in range(len(state[0])):
                    if state[y][x] != 0:  # 既に石がある
                        mask[y * len(state[0]) + x] = float('-inf')
        
        return mask
