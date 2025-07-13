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
# matplotlibバックエンドを非インタラクティブに設定（Tkinterエラーを回避）
import matplotlib
matplotlib.use('Agg')  # GUIを使用しないバックエンド
import matplotlib.pyplot as plt  # グラフ作成用にmatplotlibをインポート
from torch.optim.lr_scheduler import StepLR, ExponentialLR  # 学習率スケジューラをインポート
import numba
from numba import jit
import concurrent.futures
from functools import lru_cache
import threading  # スレッドロック用に追加
import argparse  # コマンドライン引数のパーサーを追加

# ゲーム設定
DEFAULT_BOARD_SIZE = 8  # デフォルトのボードサイズ（標準的な五目並べ）
SIMULATIONS = 200  # MCTSのシミュレーション回数（軽量化）

# トレーニング設定
DEFAULT_ITERATIONS = 5000  # デフォルトのトレーニングイテレーション数（現実的な値に）
DEFAULT_SELF_PLAY_GAMES = 32  # イテレーションごとの自己対戦ゲーム数（適度な値に）
DEFAULT_LEARNING_RATE = 0.001  # 初期学習率（少し増加）

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def value_to_class_label(value):
    """
    ゲーム結果の値(-1, 0, 1)を3クラスのラベル(0, 1, 2)に変換
    -1 (敗北) -> 0
    0 (引き分け) -> 1  
    1 (勝利) -> 2
    """
    if value < -0.5:
        return 0  # 敗北
    elif value > 0.5:
        return 2  # 勝利
    else:
        return 1  # 引き分け

def class_label_to_value(label):
    """
    3クラスのラベル(0, 1, 2)をゲーム結果の値(-1, 0, 1)に変換
    0 -> -1 (敗北)
    1 -> 0 (引き分け)
    2 -> 1 (勝利)
    """
    if label == 0:
        return -1.0
    elif label == 2:
        return 1.0
    else:
        return 0.0

class DualNetwork(nn.Module):
    """方策と価値を出力するニューラルネットワーク"""
    def __init__(self, board_size, num_channels=32, num_res_blocks=3):  # 初期チャネル数を32に変更、残差ブロック数を減らす
        super(DualNetwork, self).__init__()
        self.board_size = board_size
        
        # 入力層 (4チャネル入力に変更)
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
        
        # 方策ヘッド (Kerasモデルに合わせて4フィルタの畳み込み層を使用)
        self.policy_conv = nn.Conv2d(128, 4, 1, stride=1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)
        
        # 価値ヘッド (回帰用：連続値出力)
        self.value_conv = nn.Conv2d(128, 2, 1, stride=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.value_dropout = nn.Dropout(0.3)
        self.value_fc2 = nn.Linear(64, 1)  # 1次元出力に変更（回帰）
    
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
        # log_softmaxは損失計算時に適用するため、ここでは生のlogitsを返す
        
        # 価値ヘッド
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 2 * self.board_size * self.board_size)
        value = self.relu(self.value_fc1(value))
        value = self.value_dropout(value)
        value_logits = self.value_fc2(value)  # 回帰用の連続値出力
        
        return policy_logits, value_logits

class MCTSNode:
    """モンテカルロ木探索のノード"""
    def __init__(self, prior=0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.win_count = 0  # 勝利数を追跡
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def win_rate(self):
        """勝率（w/n）を計算"""
        if self.visit_count == 0:
            return 0
        return self.win_count / self.visit_count

class MCTS:
    """モンテカルロ木探索の実装 - 軽量化バージョン"""
    def __init__(self, model, num_simulations=200, c_puct=4.0, use_gumbel=False, gumbel_scale=0.1, 
                 add_root_noise=True, dirichlet_alpha=0.15, dirichlet_weight=0.25,
                 temperature_threshold=30, initial_temperature=1.0, final_temperature=1e-3):  # シミュレーション回数を削減
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.board_size = model.board_size
        self.use_gumbel = use_gumbel
        self.gumbel_scale = gumbel_scale
        # ルートノイズ関連のパラメータを追加
        self.add_root_noise = add_root_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        # ボルツマン分布の温度パラメータ
        self.temperature_threshold = temperature_threshold  # 温度切り替えの手数閾値
        self.initial_temperature = initial_temperature      # 序盤の温度
        self.final_temperature = final_temperature          # 終盤の温度
        # スレッドロックを追加
        self.lock = threading.Lock()

    # 空きマスを取得
    def get_valid_moves(self, state):
        valid_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if state[y][x] == 0:
                    valid_moves.append(y * self.board_size + x)
        return valid_moves
    
    # 勝利パターンを検出（一手で勝てるか）
    def detect_winning_move(self, state, player):
        valid_moves = self.get_valid_moves(state)
        for move in valid_moves:
            y, x = divmod(move, self.board_size)
            # 一時的に石を置いてみる
            state[y][x] = player
            
            # 勝利条件チェック（5つ並ぶかどうか）
            if self.check_win(state, x, y, player):
                # 元に戻す
                state[y][x] = 0
                return move
            
            # 元に戻す
            state[y][x] = 0
        
        return None

    # 相手の勝利を阻止する手を検出
    def detect_blocking_move(self, state, player):
        # 相手のプレイヤー番号
        opponent = -player
        
        # 相手が次の手で勝てる場所を検出
        return self.detect_winning_move(state, opponent)
        
    # 勝利条件チェック
    def check_win(self, state, x, y, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、縦、右下がり斜め、右上がり斜め
        
        for dx, dy in directions:
            count = 1  # 自分自身
            
            # 正方向
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and state[ny][nx] == player:
                count += 1
                nx, ny = nx + dx, ny + dy
                
            # 逆方向
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and state[ny][nx] == player:
                count += 1
                nx, ny = nx - dx, ny - dy
                
            if count >= 5:
                return True
                
        return False

    def _ucb_score(self, parent, child, action):
        """UCB (Upper Confidence Bound) スコアの計算"""
        prior_score = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        
        # 勝率（w/n）を計算
        if child.visit_count > 0:
            win_rate = child.win_rate()
            value_score = win_rate  # 勝率を直接使用
        else:
            value_score = 0
        
        # 勝ち確定手なら無限大のスコア（この機能を強化）
        if parent.state is not None:
            # 勝利パターンの検出
            y, x = divmod(action, self.board_size)
            temp_state = parent.state.copy()
            player = 1 if np.sum(temp_state == 1) == np.sum(temp_state == -1) else -1
            temp_state[y][x] = player
            
            # 直接勝利する手は最優先
            if self.check_win(temp_state, x, y, player):
                return float('inf')
                
            # 元の判定も残す
            if self._is_winning_move(parent.state, action) or self._has_open_four(parent.state, action):
                return float('inf')
        
        score = value_score + prior_score
        
        # # Gumbelノイズを追加（有効な場合）
        # if self.use_gumbel:
        #     # Gumbel(0, scale)分布からノイズを生成
        #     noise = np.random.gumbel(0, self.gumbel_scale)
        #     score += noise
                
        return score
    
    def _has_open_four(self, state, move):
        """四つ並んでいて相手が干渉していない状況を検出する（安全で効率的な実装）"""
        x, y = move % self.board_size, move // self.board_size
        if state[y][x] != 0:  # すでに石が置かれている場合
            return False
        
        # プレイヤーを特定
        player = 1 if np.sum(state == 1) == np.sum(state == -1) else -1
        
        # 安全に操作するためのコピーを作成
        temp_state = state.copy()
        temp_state[y][x] = player
        
        # 方向ベクトル定義: 横、縦、右下、左下
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # 自分自身
            
            # 両端の開放状態を確認
            pos_open = False
            neg_open = False
            
            # 正方向に連続する石を数える
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and temp_state[ny][nx] == player:
                count += 1
                nx += dx
                ny += dy
            
            # 正方向の端が空いているか確認
            pos_open = (0 <= nx < self.board_size and 0 <= ny < self.board_size and temp_state[ny][nx] == 0)
            
            # 負方向に連続する石を数える
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and temp_state[ny][nx] == player:
                count += 1
                nx -= dx
                ny -= dy
            
            # 負方向の端が空いているか確認
            neg_open = (0 <= nx < self.board_size and 0 <= ny < self.board_size and temp_state[ny][nx] == 0)
            
            # 四つ並びで少なくとも片方の端が空いている場合
            if count == 4 and (pos_open or neg_open):
                return True
        
        return False
    
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
    
    def search(self, state, last_move=None):
        """与えられた状態に基づいてMCTSを実行"""
        root = MCTSNode(0)
        root.state = state.copy()
        
        # 現在のプレイヤーを特定
        current_player = 1 if np.sum(state == 1) == np.sum(state == -1) else -1
        
        # 評価関数から方策と価値を取得（キャッシュなしで直接評価）
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            # 最後の手と現在のプレイヤー情報を渡す
            policy_logits, value_logits = self.model(state_tensor, last_move, current_player)
        
        # policy_logitsは生のlogitsなので、softmaxを適用して確率分布に変換
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        
        # value_logitsから価値を取得（回帰出力なのでtanhを適用）
        value = torch.tanh(value_logits).squeeze().item()
        # 現在のプレイヤーの視点から価値を調整
        value = value * current_player
        
        policy_legal = np.zeros(self.board_size * self.board_size)
        legal_moves = self._get_legal_moves(state)
        
        # 勝利パターンと防御パターンを検出（キャッシュなしで直接検出）
        winning_move = self.detect_winning_move(state.copy(), current_player)
        blocking_move = None if winning_move is not None else self.detect_blocking_move(state.copy(), current_player)

        # 勝利手リストを初期化
        winning_moves = []
        
        # 勝利パターン検出時の処理
        if winning_move is not None:
            if random.random() < 0.01:  # 1%の確率でログ出力（デバッグ過多を防ぐ）
                print("勝利パターン検出: 勝利確定の手を選択")
            winning_moves.append(winning_move)
            policy_legal = np.zeros(self.board_size * self.board_size)
            policy_legal[winning_move] = 1.0
            return policy_legal
            
        # 負け防止パターン検出時の処理
        if blocking_move is not None:
            if random.random() < 0.01:  # 1%の確率でログ出力
                print("敗北パターン検出: ブロック手を選択")
            policy_legal = np.zeros(self.board_size * self.board_size)
            policy_legal[blocking_move] = 1.0
            return policy_legal
        
        # 上記のどれにも該当しない場合は通常のMCTSで探索
        for move in legal_moves:
            policy_legal[move] = policy[move]
        
        # 合法手がある場合は正規化
        if len(legal_moves) > 0:
            policy_legal = policy_legal / np.sum(policy_legal)
        
        # ルートノードにディリクレノイズを適用（自己対戦時の探索の多様性向上）
        if self.add_root_noise and len(legal_moves) > 0:
            # ディリクレ分布からノイズを生成
            dirichlet_noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            
            # 方策にノイズを適用
            for i, move in enumerate(legal_moves):
                # 元の方策とノイズを重み付き平均で混合
                original_prob = policy_legal[move]
                noisy_prob = (1 - self.dirichlet_weight) * original_prob + self.dirichlet_weight * dirichlet_noise[i]
                policy_legal[move] = noisy_prob
                
            # 再正規化
            if np.sum(policy_legal) > 0:
                policy_legal = policy_legal / np.sum(policy_legal)
        
        # 子ノードの初期化（ノイズ適用後の方策を使用）
        for move in legal_moves:
            root.children[move] = MCTSNode(policy_legal[move])

        # 並列シミュレーション実行
        self._run_simulations_parallel(root, state.copy(), self.num_simulations)
        
        # 論文の式に基づく方策計算: π(s_0, a) ∝ N(s_0, a)^{1/τ}
        visit_counts = np.zeros(self.board_size * self.board_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # 勝ち確定手がある場合は、MCTSの探索結果に関わらず勝ち確定手を選択
        if winning_moves:
            mcts_policy = np.zeros(self.board_size * self.board_size)
            winning_move = random.choice(winning_moves)
            mcts_policy[winning_move] = 1.0
            mcts_value = 1.0
            return mcts_policy, mcts_value
        
        # 現在の手数に基づいて温度パラメータを決定（ボルツマン分布用）
        move_count = self._get_move_count(state)
        # 序盤は高い温度（探索的）、終盤は低い温度（貪欲的）
        temperature = self.initial_temperature if move_count < self.temperature_threshold else self.final_temperature
        
        # ボルツマン分布を使用して訪問回数から方策を計算
        # π(s_0, a) ∝ exp(N(s_0, a)/τ) （ボルツマン分布）
        mcts_policy = self._boltzmann_policy(visit_counts, temperature)
        
        # MCTSの価値計算を修正：ルート価値をそのまま使用
        mcts_value = root.value()  # ルートノードの価値を使用
        
        return mcts_policy, mcts_value

    def _detect_critical_moves_parallel(self, state, player):
        """勝利パターンと防御パターンを並列に検出"""
        # 並列実行を使用して両方のパターンを同時に検出
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            winning_future = executor.submit(self.detect_winning_move, state.copy(), player)
            blocking_future = executor.submit(self.detect_blocking_move, state.copy(), player)
            
            # 結果を取得
            winning_move = winning_future.result()
            # 勝利手があれば防御手はチェック不要
            blocking_move = None if winning_move is not None else blocking_future.result()
            
        return winning_move, blocking_move

    def _get_move_count(self, state):
        """盤面に置かれた石の数（着手数）を数える"""
        # 0でない要素（石が置かれたマス）の数を数える
        return np.count_nonzero(state)
    
    def _augment_data(self, state, policy):
        """データ拡張：回転と反転による盤面とポリシーの対称変換を行う"""
        # 状態とポリシーを盤面形式に変形
        board_size = self.board_size
        policy_grid = policy.reshape(board_size, board_size)
        
        # 変換されたデータを格納するリスト
        augmented_states = []
        augmented_policies = []
        
        # 元の状態とポリシーを追加
        augmented_states.append(state.copy())
        augmented_policies.append(policy.copy())
        
        # 90度回転
        rot90_state = np.rot90(state, k=1)
        rot90_policy = np.rot90(policy_grid, k=1).flatten()
        augmented_states.append(rot90_state)
        augmented_policies.append(rot90_policy)
        
        # 180度回転
        rot180_state = np.rot90(state, k=2)
        rot180_policy = np.rot90(policy_grid, k=2).flatten()
        augmented_states.append(rot180_state)
        augmented_policies.append(rot180_policy)
        
        # 270度回転
        rot270_state = np.rot90(state, k=3)
        rot270_policy = np.rot90(policy_grid, k=3).flatten()
        augmented_states.append(rot270_state)
        augmented_policies.append(rot270_policy)
        
        # 水平反転
        flip_h_state = np.fliplr(state)
        flip_h_policy = np.fliplr(policy_grid).flatten()
        augmented_states.append(flip_h_state)
        augmented_policies.append(flip_h_policy)
        
        # 垂直反転
        flip_v_state = np.flipud(state)
        flip_v_policy = np.flipud(policy_grid).flatten()
        augmented_states.append(flip_v_state)
        augmented_policies.append(flip_v_policy)
        
        # 対角線反転（左上から右下）
        diag_state = np.transpose(state)
        diag_policy = np.transpose(policy_grid).flatten()
        augmented_states.append(diag_state)
        augmented_policies.append(diag_policy)
        
        # 反対角線反転（右上から左下）
        anti_diag_state = np.transpose(np.fliplr(np.flipud(state)))
        anti_diag_policy = np.transpose(np.fliplr(np.flipud(policy_grid))).flatten()
        augmented_states.append(anti_diag_state)
        augmented_policies.append(anti_diag_policy)
        
        return augmented_states, augmented_policies

    # NumbaによるJITコンパイルを使用して高速化
    @staticmethod
    @jit(nopython=True, cache=True)
    def _check_win_pattern_numba(state, x, y, player, board_size):
        """勝利条件チェックのNumba最適化版"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # 自分自身
            
            # 正方向
            for i in range(1, 5):  # 最大4つ先まで確認
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < board_size and 0 <= ny < board_size and state[ny][nx] == player:
                    count += 1
                else:
                    break
                    
            # 負方向
            for i in range(1, 5):  # 最大4つ先まで確認
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < board_size and 0 <= ny < board_size and state[ny][nx] == player:
                    count += 1
                else:
                    break
                
            if count >= 5:
                return True
                
        return False
    
    # 勝利パターン検出のラッパー関数
    def check_win(self, state, x, y, player):
        """Numbaバージョンを使用する勝利条件チェックのラッパ"""
        # NumbaのJITコンパイル関数を呼び出す
        return self._check_win_pattern_numba(state, x, y, player, self.board_size)

    # シミュレーションの並列実行を実装
    def _run_simulations_parallel(self, root, state, num_simulations):
        """MCTSのシミュレーションを並列実行"""
        # シミュレーション数が少ない場合はシーケンシャル実行
        if num_simulations <= 16:
            for _ in range(num_simulations):
                self._run_single_simulation(root, state.copy())
            return
            
        # シミュレーションをバッチに分割して並列実行
        # batch_sizeが0にならないように最小値を1に設定
        cpu_count = max(1, mp.cpu_count())
        batch_size = max(1, min(16, num_simulations // cpu_count))
        num_batches = num_simulations // batch_size
        
        # バッチ数が0の場合はシーケンシャル実行にフォールバック
        if num_batches == 0:
            for _ in range(num_simulations):
                self._run_single_simulation(root, state.copy())
            return
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
            futures = []
            for _ in range(num_batches):
                futures.append(executor.submit(self._run_batch_simulations, root, state.copy(), batch_size))
                
            # 残りのシミュレーション
            remainder = num_simulations % batch_size
            if remainder > 0:
                futures.append(executor.submit(self._run_batch_simulations, root, state.copy(), remainder))
                
            # 全てのバッチの完了を待機
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"シミュレーション実行中にエラーが発生: {e}")

    def _run_batch_simulations(self, root, state, batch_size):
        """バッチでシミュレーションを実行"""
        for _ in range(batch_size):
            self._run_single_simulation(root, state.copy())
            
    def _run_single_simulation(self, node, state):
        """単一のMCTSシミュレーションを実行"""
        search_path = [node]
        current_state = state.copy()
        last_move = None  # 最後の手を追跡
        
        # 葉ノードを見つける
        while node.expanded():
            action, node = self._select_child(node)
            x, y = action % self.board_size, action // self.board_size
            last_move = (x, y)  # 最後の手を更新
            current_state[y][x] = -1 if np.sum(current_state == 1) > np.sum(current_state == -1) else 1
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
            
            # ニューラルネットワークで評価（最後の手と現在のプレイヤー情報を渡す）
            leaf_tensor = torch.tensor(leaf_state, dtype=torch.float32, device=device).unsqueeze(0)
            current_player = 1 if np.sum(leaf_state == 1) == np.sum(leaf_state == -1) else -1
            with torch.no_grad():
                policy_logits, value_logits = self.model(leaf_tensor, last_move, current_player)
            
            # policy_logitsは生のlogitsなので、softmaxを適用して確率分布に変換
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            
            # value_logitsをtanhに変換し、期待値を計算
            value_probs = F.tanh(value_logits).squeeze(0)  # [3]
            # 各クラス（敗北:-1, 引き分け:0, 勝利:1）の期待値を計算
            value = -1.0 * value_probs[0] + 0.0 * value_probs[1] + 1.0 * value_probs[2]
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
            # 勝利数を更新（価値が正の場合は勝利とみなす）
            if value > 0:
                node.win_count += 1
            value = -value  # 交互に手番が変わるので、価値を反転

    def _select_child(self, node):
        """UCBスコアに基づいて子ノードを選択"""
        # UCBスコアが最大の行動と対応する子ノードを選択
        best_score = float('-inf')
        best_action = -1
        best_child = None
        
        # スレッドセーフにするため、反復処理前に子ノードのリストをコピー
        with self.lock:
            items = list(node.children.items())
        random.shuffle(items)

        # コピーした項目に対してUCBスコアを計算
        for action, child in items:
            score = self._ucb_score(node, child, action)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _is_terminal(self, state):
        """ゲームが終了状態かどうかをチェック"""
        # 勝者がいる場合は終了状態
        if self._get_winner_value(state) != 0:
            return True
            
        # 盤面に空きマスがない場合も終了状態（引き分け）
        for y in range(self.board_size):
            for x in range(self.board_size):
                if state[y][x] == 0:
                    return False
        
        return True
    
    # _get_legal_movesの最適化版
    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_legal_moves(state):
        """盤面の合法手を返すNumba最適化版"""
        board_size = state.shape[0]
        legal_moves = []
        for y in range(board_size):
            for x in range(board_size):
                if state[y][x] == 0:
                    legal_moves.append(y * board_size + x)
        return legal_moves
    
    def _boltzmann_policy(self, visit_counts, temperature):
        """ボルツマン分布を使用して訪問回数から方策を計算
        
        Args:
            visit_counts: 各行動の訪問回数配列
            temperature: 温度パラメータ（高いほど探索的、低いほど貪欲）
            
        Returns:
            ボルツマン分布に基づく方策配列
        """
        # 温度がほぼゼロの場合はグリーディー選択（デルタ関数）
        if temperature < 1e-6:
            action = np.argmax(visit_counts)
            policy = np.zeros(len(visit_counts))
            policy[action] = 1.0
            return policy
        
        # 訪問回数がゼロでないアクションのみを考慮
        nonzero_indices = np.where(visit_counts > 0)[0]
        if len(nonzero_indices) == 0:
            # すべての訪問回数がゼロの場合は一様分布
            return np.ones(len(visit_counts)) / len(visit_counts)
        
        try:
            # ボルツマン分布: P(a) ∝ exp(N(a)/τ)
            # ここでN(a)は訪問回数、τは温度パラメータ
            counts = visit_counts[nonzero_indices]
            
            # 数値安定性のために対数を使用してソフトマックスを計算
            # log(P(a)) = N(a)/τ - log(Σ exp(N(b)/τ))
            logits = counts / temperature
            
            # 数値安定性のために最大値を引く（log-sum-exp trick）
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            
            # 正規化してボルツマン確率を計算
            boltzmann_probs = exp_logits / np.sum(exp_logits)
            
            # 方策の初期化（全体）
            policy = np.zeros(len(visit_counts))
            # 非ゼロ訪問回数の箇所にのみボルツマン確率を設定
            policy[nonzero_indices] = boltzmann_probs
            
            return policy
            
        except Exception as e:
            # エラーが発生した場合は最も訪問回数の多い行動を選択
            print(f"ボルツマン分布計算中にエラーが発生: {e}")
            action = np.argmax(visit_counts)
            policy = np.zeros(len(visit_counts))
            policy[action] = 1.0
            return policy
        
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
    def __init__(self, capacity=40000):
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
    mcts = MCTS(model, num_simulations=SIMULATIONS, use_gumbel=True, gumbel_scale=0.05)  # 訓練時も計算量を増加
    
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
        
        # 行動の選択（ε-greedy探索を削除し、常にMCTSの方策に従う）
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

def train_network(model, replay_buffer, epochs=10, batch_size=256, lr=0.001):
    """ニューラルネットワークの訓練 - 損失関数修正版"""
    model.train()
    
    # AdamWオプティマイザーを使用（より安定したweight decay）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    # Cosine Annealing学習率スケジューラーを使用（より安定）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
    
    # リプレイバッファからデータを取得
    if len(replay_buffer) < batch_size:
        return 0, 0  # データが十分でない場合はスキップ
    
    # サンプル数を制限して訓練を高速化
    max_samples = min(len(replay_buffer), 3000)  # サンプル数上限を削減
    states, policies, values = replay_buffer.sample(max_samples)
    
    # 価値の前処理：範囲を[-1, 1]にクリップ
    values = np.clip(values, -1.0, 1.0)
    
    # デバッグ：価値の統計を出力
    print(f"価値の統計 - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}, Min: {np.min(values):.4f}, Max: {np.max(values):.4f}")
    
    # データセットとデータローダーの作成
    dataset = SelfPlayDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 早期停止のための変数
    best_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    # 損失を記録する変数を初期化
    total_policy_loss = 0
    total_value_loss = 0
    
    # AlphaZero論文に従った正則化重み設定
    l2_reg_weight = 1e-4
    
    for epoch in range(epochs):
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_total_loss = 0
        batch_count = 0
        
        for batch_states, batch_policies, batch_values in dataloader:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device).view(-1)  # バッチサイズに合わせて形状調整
            
            # 予測
            policy_logits, value_logits = model(batch_states)
            # value_logitsは[batch_size, 1]の形状（回帰）
            
            # AlphaZero論文に従った損失関数の実装
            # Policy Loss: クロスエントロピー損失 = -Σ(πᵢ * log(pᵢ))
            # マスクを適用して不正な手への確率を0にする
            # 有効な手のマスクを作成
            valid_actions_mask = torch.ones_like(policy_logits, dtype=torch.bool)
            for i, state in enumerate(batch_states):
                # 空でないセルは無効な手とする
                board_state = state[0] if state.dim() == 3 else state.squeeze(0)
                invalid_positions = (board_state != 0).view(-1)
                valid_actions_mask[i][invalid_positions] = False
            
            # マスクされた方策に対してsoftmaxを適用
            masked_policy_logits = policy_logits.clone()
            masked_policy_logits[~valid_actions_mask] = -float('inf')
            policy_probs = F.softmax(masked_policy_logits, dim=1)
            
            # KLダイバージェンスを使用した方策損失（より安定）
            # 小さな値を加えて数値的安定性を向上
            epsilon = 1e-8
            policy_probs = policy_probs + epsilon
            batch_policies = batch_policies + epsilon
            
            # 正規化
            policy_probs = policy_probs / policy_probs.sum(dim=1, keepdim=True)
            batch_policies = batch_policies / batch_policies.sum(dim=1, keepdim=True)
            
            policy_loss = F.kl_div(torch.log(policy_probs), batch_policies, reduction='batchmean')
            
            # Value Loss: 平均二乗誤差損失（AlphaZero論文に従う）
            # value_logitsを[-1, 1]の範囲にスケールするためtanhを適用
            value_predictions = torch.tanh(value_logits.squeeze(-1))
            value_loss = F.mse_loss(value_predictions, batch_values)
            
            # L2正則化項（重みパラメータのみに適用、AlphaZero論文に従う）
            l2_reg = 0
            for name, param in model.named_parameters():
                if 'weight' in name:  # バイアス項は除外し、重みのみに正則化を適用
                    l2_reg += torch.norm(param, p=2)
            
            # 総損失の計算（AlphaZero論文の式に従う）
            # L = c1 * value_loss + c2 * policy_loss + c3 * ||θ||²
            # 重み付けを調整（価値損失により重点を置く）
            value_loss_weight = 1.0  # 価値損失の重み
            policy_loss_weight = 1.0  # 方策損失の重み
            l2_reg_weight = 1e-4  # L2正則化の重み
            
            total_loss = (value_loss_weight * value_loss + 
                         policy_loss_weight * policy_loss + 
                         l2_reg_weight * l2_reg)
            
            # デバッグ：バッチごとの損失を出力（最初のエポックの最初の3バッチのみ）
            if epoch == 0 and batch_count < 3:
                print(f"Batch {batch_count}:")
                print(f"  Policy Loss: {policy_loss.item():.6f}")
                print(f"  Value Loss: {value_loss.item():.6f}")
                print(f"  L2 Reg: {l2_reg.item():.6f}")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  Predicted values: {value_predictions[:5].detach().cpu().numpy()}")
                print(f"  Target values: {batch_values[:5].detach().cpu().numpy()}")
                print(f"  Policy probs sample: {policy_probs[0][:10].detach().cpu().numpy()}")
                print(f"  Target policy sample: {batch_policies[0][:10].detach().cpu().numpy()}")
            
            # 勾配の計算と更新
            optimizer.zero_grad()
            total_loss.backward()
            
            # 勾配クリッピング（より適切な値に変更）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 勾配の統計を監視（デバッグ用）
            if epoch == 0 and batch_count < 3:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"  Gradient norm: {total_norm:.6f}")
            
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            batch_count += 1
        
        # エポック平均損失の計算
        avg_policy_loss = epoch_policy_loss / len(dataloader)
        avg_value_loss = epoch_value_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)
        
        # 早期停止チェック（総損失で判定）
        # 改善の閾値を設定（わずかな改善でも継続）
        improvement_threshold = 1e-4
        if avg_total_loss < best_loss - improvement_threshold:
            best_loss = avg_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best loss: {best_loss:.6f})")
                break
        
        # エポックごとに学習率を減衰
        scheduler.step()
        
        # エポックごとの損失を追加
        total_policy_loss += avg_policy_loss
        total_value_loss += avg_value_loss
        
        # 現在の学習率を出力（デバッグ用）
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}")
        print(f"  Policy Loss: {avg_policy_loss:.6f}, Value Loss: {avg_value_loss:.6f}")
        print(f"  Total Loss: {avg_total_loss:.6f}")
        
        # 損失が異常値になった場合の対処
        if np.isnan(avg_policy_loss) or np.isnan(avg_value_loss):
            print("警告: 損失がNaNになりました。訓練を停止します。")
            break
        
        if avg_policy_loss > 100 or avg_value_loss > 100:
            print("警告: 損失が異常に大きくなりました。学習率を下げます。")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
    
    # 完了したエポック数で平均を計算
    completed_epochs = epoch + 1
    avg_policy_loss = total_policy_loss / completed_epochs
    avg_value_loss = total_value_loss / completed_epochs
    
    return avg_policy_loss, avg_value_loss

class AlphaZero:
    """AlphaZeroの実装"""
    def __init__(self, board_size=19, num_iterations=100, num_self_play_games=64,
                 checkpoint_dir='models', log_dir='logs', initial_lr=0.0005):
        self.board_size = board_size
        self.num_iterations = num_iterations
        self.num_self_play_games = num_self_play_games
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.initial_lr = initial_lr  # 初期学習率を保存
        
        # ディレクトリの作成
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # モデルの初期化
        self.model = DualNetwork(board_size).to(device)
        
        # リプレイバッファの容量を大幅に増加（より多くのデータを保持）
        self.replay_buffer = ReplayBuffer(capacity=40000)  
        
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
        # 学習率の履歴を記録するリストを追加
        self.lr_history = []
        
        # 既存のモデルをロード
        if os.path.exists(self.model_path):
            try:
                # self.model.load_state_dict(torch.load(self.model_path))
                print(f"モデルを読み込みました: {self.model_path}")
            except:
                print("新規モデルを初期化します")
        else:
            # 最新のモデルファイルを検索
            model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(self.model_base_name) and f.endswith('.pth')]
            if model_files:
                # 最新のモデルをロード
                latest_model = os.path.join(checkpoint_dir, sorted(model_files)[-1])
                try:
                    self.model.load_state_dict(torch.load(latest_model, map_location=device))
                    print(f"最新のモデルを読み込みました: {latest_model}")
                    self.model_path = latest_model
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
            
            # イテレーションが進むにつれて初期学習率を調整
            # 例：20イテレーションごとに初期学習率を半分にする
            current_lr = self.initial_lr * (0.5 ** (iteration // 20))
            print(f"現在の初期学習率: {current_lr:.6f}")
            self.lr_history.append(current_lr)
            
            # イテレーション番号を追加（重複しないように一度だけ追加）
            current_iter = iteration + 1
            
            # 1. 自己対戦でデータ生成（並列）
            print("自己対戦でデータを生成中...")
            self._generate_self_play_data()
            
            # 自己対戦後のモデルを保存
            self.save_model("after_selfplay", iteration+1)
            
            # 2. ニューラルネットワークの訓練
            print("ニューラルネットワークを訓練中...")
            policy_loss, value_loss = train_network(self.model, self.replay_buffer, lr=current_lr)
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            # 損失履歴に追加
            if policy_loss > 0:  # 正常な損失値の場合のみ追加
                self.policy_loss_history.append(policy_loss)
                self.value_loss_history.append(value_loss)
                self.iterations.append(current_iter)
            
            # 訓練後のモデルを保存
            self.save_model("trained", iteration+1)
            
            # イテレーション完了時に1回だけグラフを保存
            self._plot_loss_history(iteration=current_iter)
            
            # 3. トレーニング情報をログファイルに保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(self.timestamp_log_dir, f'training_log_{iteration+1}_{timestamp}.txt')
            
            with open(log_filename, 'w') as f:
                f.write(f"Training Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Path: {self.model_path}\n")
                f.write(f"Iteration: {iteration+1}/{self.num_iterations}\n")
                f.write(f"Initial Learning Rate: {current_lr:.6f}\n")
                f.write(f"Policy Loss: {policy_loss:.6f}\n")
                f.write(f"Value Loss: {value_loss:.6f}\n")
                f.write(f"Replay Buffer Size: {len(self.replay_buffer)}\n")
                f.write(f"Self-Play Games: {self.num_self_play_games}\n")
                f.write(f"Board Size: {self.board_size}\n")
                f.write(f"Training Duration: {time.time() - start_time:.2f} seconds\n")
            
            print(f"トレーニング情報をログに保存しました: {log_filename}")
            
            iteration_time = time.time() - start_time
            print(f"イテレーション完了: {iteration_time:.2f} 秒")
        
        # トレーニング終了時の最終モデルを保存
        self.save_model("final")

    def _plot_loss_history(self, final=False, stage=None, iteration=None):
        """損失の履歴をグラフ化して保存"""
        plt.figure(figsize=(15, 15))  # グラフのサイズを大きくして3つのグラフを表示
        
        # グラフタイトルに情報を追加
        title_suffix = ""
        if final:
            title_suffix = " (Final)"
        elif iteration:
            title_suffix = f" (Iteration {iteration})"
        
        # 学習率の履歴データを準備
        lr_iterations = list(range(1, len(self.lr_history) + 1))
        
        # 最終的な表示領域を3つに分割
        total_subplots = 3 if len(self.policy_loss_history) > 0 else 1
        
        # Policy Lossグラフ (データがある場合のみ)
        if len(self.policy_loss_history) > 0:
            plt.subplot(total_subplots, 1, 1)  # total_subplots行1列の1番目
            plt.plot(self.iterations, self.policy_loss_history, 'b-', marker='o')
            plt.title(f'Policy Loss History{title_suffix}')
            plt.xlabel('Iteration')
            plt.ylabel('Policy Loss')
            plt.grid(True)
            
            # Value Lossグラフ
            plt.subplot(total_subplots, 1, 2)  # total_subplots行1列の2番目
            plt.plot(self.iterations, self.value_loss_history, 'r-', marker='o')
            plt.title(f'Value Loss History{title_suffix}')
            plt.xlabel('Iteration')
            plt.ylabel('Value Loss')
            plt.grid(True)
        
        # 学習率の履歴を追加 (常に表示)
        plt.subplot(total_subplots, 1, total_subplots)  # 最後の位置
        plt.plot(lr_iterations, self.lr_history, 'g-', marker='o')
        plt.title(f'Learning Rate History{title_suffix}')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.yscale('log')  # 学習率は対数スケールで表示
        plt.grid(True)
        
        # グラフ下部に現在の訓練状況を表示
        plt.figtext(0.5, 0.01, 
                   f"Board Size: {self.board_size}, Buffer Size: {len(self.replay_buffer)}, Games/Iter: {self.num_self_play_games}", 
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # ファイル名の作成（シンプル化）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if iteration:
            plot_filename = os.path.join(self.timestamp_log_dir, f'loss_history_iter{iteration}_{timestamp}.png')
        else:
            plot_filename = os.path.join(self.timestamp_log_dir, f'loss_history_final_{timestamp}.png')
        
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
        
        # 共有メモリの準備 - より効率的な方法
        manager = mp.Manager()
        shared_buffer = manager.list()
        
        # プロセス数を動的に設定（リソース使用量を調整）
        available_cpu = mp.cpu_count()
        num_processes = min(int(available_cpu * 0.6), self.num_self_play_games, 8)  # 最大4プロセスに制限
        num_processes = max(1, num_processes)  # 少なくとも1プロセス
        
        games_per_process = self.num_self_play_games // num_processes
        remainder = self.num_self_play_games % num_processes
        
        print(f"並列プロセス数: {num_processes}, プロセスあたりの基本ゲーム数: {games_per_process}")
        
        # プロセス作成と開始
        start_idx = 0
        for i in range(num_processes):
            # 残りのゲームを均等に分配
            process_games = games_per_process + (1 if i < remainder else 0)
            end_idx = start_idx + process_games
            
            p = ctx.Process(
                target=self._self_play_process,
                args=(self.model_path, self.board_size, shared_buffer, range(start_idx, end_idx), result_queue)
            )
            p.start()
            processes.append(p)
            start_idx = end_idx
        
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
        
        # リプレイバッファにデータを追加（バッチ処理で効率化)
        batch_size = 1000
        buffer_list = list(shared_buffer)
        for i in range(0, len(buffer_list), batch_size):
            batch = buffer_list[i:i+batch_size]
            for state, policy, value in batch:
                self.replay_buffer.add(state, policy, value)
        
        print(f"リプレイバッファサイズ: {len(self.replay_buffer)}")

    def _self_play_process(self, model_path, board_size, shared_buffer, game_indices, result_queue):
        """自己対戦プロセス（価値計算修正版）"""
        # モデルのロード
        model = DualNetwork(board_size)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"モデルロードエラー: {e}")
            # 初期ランダムモデルで続行
        
        model.to(device)
        model.eval()
        
        # 基本シミュレーション回数の設定（最小値を確保）
        base_simulations = max(50, SIMULATIONS)  # 最低50回のシミュレーションを保証
        total_spaces = board_size * board_size
        mcts = MCTS(model, num_simulations=base_simulations, use_gumbel=True, gumbel_scale=0.02)
        
        # 一時的なバッファ（メモリ効率向上）
        temp_buffer = []
        max_temp_size = 2000  # 一時的なバッファの最大サイズを増加（1000から2000に）
        
        for game_idx in game_indices:
            try:
                # 環境の初期化
                env = GomokuEnv(board_size=board_size)
                
                # MCTSの動的調整のために状態を追跡
                current_state = env.board.GetBoardInt()
                
                # 残り空きマスの数に応じてシミュレーション回数を調整する関数
                def adjust_simulations(state):
                    remaining_spaces = np.sum(np.array(state) == 0)
                    ratio = remaining_spaces / total_spaces
                    # 最低でも30回、最大でも基本シミュレーション回数を確保
                    adjusted_sims = max(30, min(int(base_simulations * (0.5 + 0.5 * ratio)), base_simulations))
                    return adjusted_sims
                
                # 初期シミュレーション回数を設定
                mcts.num_simulations = base_simulations
                
                game_memory = []
                state = env.board.GetBoardInt()
                
                done = False
                current_player = 1
                move_count = 0
                last_move = None  # 最後の手を追跡
                
                # ゲーム実行
                while not done and move_count < board_size * board_size:
                    move_count += 1
                    state_array = np.array(state)
                    
                    # シミュレーション回数を現在の盤面状態に基づいて調整
                    adjusted_sims = adjust_simulations(state_array)
                    mcts.num_simulations = adjusted_sims
                    
                    # 現在のプレイヤーを特定
                    player = 1 if np.sum(state_array == 1) == np.sum(state_array == -1) else -1
                    
                    # 勝利パターンチェック - 勝利確定なら優先
                    winning_move = mcts.detect_winning_move(state_array.copy(), player)
                    if winning_move is not None:
                        # 勝利手が見つかった場合、その手を選ぶ
                        one_hot = np.zeros(board_size * board_size)
                        one_hot[winning_move] = 1.0
                        mcts_policy = one_hot
                        mcts_value = 1.0  # 勝利確定なので価値は1.0
                    else:
                        # 負け防止パターンチェック
                        blocking_move = mcts.detect_blocking_move(state_array.copy(), player)
                        if blocking_move is not None:
                            # ブロック手を選択
                            one_hot = np.zeros(board_size * board_size)
                            one_hot[blocking_move] = 1.0
                            mcts_policy = one_hot
                            mcts_value = 0.0  # 防御手なので中立的な価値
                        else:
                            # 通常のMCTSで方策と価値を取得（調整済みのシミュレーション回数を使用）
                            try:
                                mcts_policy, mcts_value = mcts.search(state_array, last_move)
                            except Exception as e:
                                print(f"MCTS検索中にエラーが発生: {e}")
                               
                                # フォールバック: ランダム方策と中立価値
                                legal_moves = [(i % board_size, i // board_size) 
                                              for i in range(board_size * board_size) 
                                              if state_array[i // board_size][i % board_size] == 0]
                                if legal_moves:
                                    mcts_policy = np.zeros(board_size * board_size)
                                    for move in legal_moves:
                                        idx = move[1] * board_size + move[0]
                                        mcts_policy[idx] = 1.0 / len(legal_moves)
                                    mcts_value = 0.0
                                else:
                                    break  # 合法手がない場合はゲーム終了
                    
                    # 訓練データの保存（プレイヤー視点での価値を保存）
                    game_memory.append((state_array.copy(), mcts_policy.copy(), mcts_value, player))
                    
                    # 行動の選択（ε-greedy探索を削除し、常にMCTSの方策に従う）
                    # 方策に従った行動選択（論文の温度パラメータに基づく）
                    try:
                        action_idx = np.random.choice(len(mcts_policy), p=mcts_policy)
                        action = (action_idx % board_size, action_idx // board_size)
                    except Exception as e:
                        print(f"行動選択中にエラーが発生: {e}")
                        # フォールバック: ランダム選択
                        legal_moves = [(i % board_size, i // board_size) 
                                      for i in range(board_size * board_size) 
                                      if state_array[i // board_size][i % board_size] == 0]
                        if legal_moves:
                            action = random.choice(legal_moves)
                        else:
                            break
                    
                    # 環境での行動実行
                    try:
                        next_state, reward, done, _ = env.step(action)
                        last_move = action  # 最後の手を更新
                        state = next_state.cpu().numpy()
                        current_player *= -1  # プレイヤー交代
                    except Exception as e:
                        print(f"環境ステップ実行中にエラーが発生: {e}")
                        break
                
                # ゲーム終了後の処理
                final_game_result = reward.item() if 'reward' in locals() else 0
                
                # 価値の範囲チェックとデバッグ出力
                value_samples = []
                
                # 各手のデータに対して処理（ゲーム結果を反映）
                for i, (hist_state, hist_policy, hist_mcts_value, hist_player) in enumerate(game_memory):
                    # ゲーム結果に基づく最終価値を計算
                    if final_game_result == 0:  # 引き分け
                        final_value = 0.0
                    else:
                        # プレイヤー視点でのゲーム結果を計算
                        final_value = final_game_result if hist_player == 1 else -final_game_result
                    
                    # MCTSの価値と最終結果を重み付け平均で組み合わせ
                    # ゲーム終盤に近いほど最終結果の重みを大きくする（改良版）
                    game_progress = i / len(game_memory)  # 0～1の進行度
                    result_weight = 0.2 + 0.6 * game_progress  # 0.2～0.8の重み（より緩やか）
                    mcts_weight = 1.0 - result_weight
                    
                    training_value = mcts_weight * hist_mcts_value + result_weight * final_value
                    
                    # 価値を[-1, 1]の範囲にクリップ
                    training_value = np.clip(training_value, -1.0, 1.0)
                    
                    # デバッグ用のサンプリング
                    if len(value_samples) < 5:
                        value_samples.append(training_value)
                    
                    # データ拡張率を動的に調整（より保守的に）
                    aug_prob = max(0.1, 0.3 - (len(shared_buffer) / 50000))
                    
                    if np.random.random() < aug_prob:
                        # 対称性を活用してデータを拡張
                        try:
                            augmented_states, augmented_policies = mcts._augment_data(hist_state, hist_policy)
                            # 拡張データを制限（2つまで）
                            for i, (aug_state, aug_policy) in enumerate(zip(augmented_states[:3], augmented_policies[:3])):
                                temp_buffer.append((aug_state, aug_policy, training_value))
                        except Exception as e:
                            print(f"データ拡張中にエラーが発生: {e}")
                            # 拡張に失敗した場合はオリジナルデータのみ追加
                            temp_buffer.append((hist_state, hist_policy, training_value))
                    else:
                        # 拡張せずオリジナルのデータのみ追加
                        temp_buffer.append((hist_state, hist_policy, training_value))
                    
                    # 一時バッファが一定サイズを超えたらshared_bufferに転送
                    if len(temp_buffer) >= max_temp_size:
                        try:
                            for item in temp_buffer:
                                shared_buffer.append(item)
                            temp_buffer = []
                        except Exception as e:
                            print(f"バッファ転送中にエラーが発生: {e}")
                            temp_buffer = []  # エラーが発生した場合はバッファをクリア
                
                # デバッグ出力（1ゲームごと）
                if len(value_samples) > 0:
                    print(f"ゲーム{game_idx}: 最終結果={final_game_result}, "
                          f"価値サンプル={[f'{v:.3f}' for v in value_samples]}")
                
                # 結果をキューに送信
                result_queue.put((game_idx, final_game_result))
                
            except Exception as e:
                print(f"ゲーム{game_idx}実行中にエラーが発生: {e}")
                # エラーが発生した場合でも結果を送信してデッドロックを防ぐ
                result_queue.put((game_idx, 0))
        
        # 残りのデータをshared_bufferに転送
        try:
            for item in temp_buffer:
                shared_buffer.append(item)
        except Exception as e:
            print(f"最終バッファ転送中にエラーが発生: {e}")

    def play_against_human(self):
        """人間との対戦"""
        env = GomokuEnv(board_size=self.board_size)
        state = env.board.GetBoardInt()
        
        # MCTSの初期化 (対戦時はシミュレーション回数を適度に)
        base_simulations = 2000
        total_spaces = self.board_size * self.board_size
        mcts = MCTS(self.model, num_simulations=base_simulations, use_gumbel=False)
        
        # 残り空きマスの数に応じてシミュレーション回数を調整する関数
        def adjust_simulations(state):
            remaining_spaces = np.sum(np.array(state) == 0)
            ratio = remaining_spaces / total_spaces
            # 最低でも基本シミュレーション数の20%は確保、最大は基本シミュレーション回数
            adjusted_sims = max(int(base_simulations * ratio), int(base_simulations * 0.2))
            return adjusted_sims
        
        done = False
        human_first = input("先手で始めますか？ (y/n): ").lower() == 'y'
        
        if not human_first:
            # AIの手番
            state_array = np.array(state)
            
            # シミュレーション回数を調整
            mcts.num_simulations = adjust_simulations(state_array)
            
            current_player = 1 if np.sum(state_array == 1) == np.sum(state_array == -1) else -1
            
            # 勝利パターンチェック
            winning_move = mcts.detect_winning_move(state_array.copy(), current_player)
            if winning_move is not None:
                print("AIが勝利パターンを検出しました")
                action_idx = winning_move
            else:
                # 負け防止パターンチェック
                blocking_move = mcts.detect_blocking_move(state_array.copy(), current_player)
                if blocking_move is not None:
                    print("AIが負け防止パターンを検出しました")
                    action_idx = blocking_move
                else:
                    # 通常のMCTSで手を決定
                    mcts_policy = mcts.search(state_array)
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
                state_array = np.array(state)
                
                # シミュレーション回数を現在の盤面状態に基づいて調整
                mcts.num_simulations = adjust_simulations(state_array)
                
                current_player = 1 if np.sum(state_array == 1) == np.sum(state_array == -1) else -1
                
                # 勝利パターンチェック
                winning_move = mcts.detect_winning_move(state_array.copy(), current_player)
                if winning_move is not None:
                    print("AIが勝利パターンを検出しました")
                    action_idx = winning_move
                    
                    # 勝利確定手を実行
                    action = (action_idx % self.board_size, action_idx // self.board_size)
                    state, reward, done, _ = env.step(action)
                    state = state.cpu().numpy()
                    env.render()
                    
                    if done:
                        print("AIの勝利です！")
                    break  # 勝利確定なのでゲームを終了
                else:
                    # 負け防止パターンチェック
                    blocking_move = mcts.detect_blocking_move(state_array.copy(), current_player)
                    if blocking_move is not None:
                        print("AIが負け防止パターンを検出しました")
                        action_idx = blocking_move
                    else:
                        # 通常のMCTSで手を決定
                        mcts_policy = mcts.search(state_array)
                        action_idx = np.argmax(mcts_policy)
                
                action = (action_idx % self.board_size, action_idx // self.board_size)
                state, reward, done, _ = env.step(action)
                state = state.cpu().numpy()
                env.render()
                
                if done:
                    print("AIの勝利です！")
                
            except ValueError:
                print("数値を入力してください")
                continue
    
    def _evaluate_neighbors(self, state, x, y, player):
        """候補手の周囲の石の配置に基づいて評価値を計算する高速な関数"""
        score = 0
        directions = [(1, 0
                       ), (0, 1), (1, 1), (1, -1)]  # 横、縦、右下がり斜め、右上がり斜め
        
        # 周辺の自分の石と相手の石を評価
        for d in range(1, 3):  # 1~2マス範囲をチェック
            for dx, dy in directions:
                # 正方向
                nx, ny = x + d*dx, y + d*dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if state[ny][nx] == player:  # 自分の石
                        score += 1.0 / (d + 1)   # 近いほど高スコア
                    elif state[ny][nx] == -player:  # 相手の石
                        score += 0.5 / (d + 1)   # 相手の石に対しても少しスコアを与える（ブロック価値）
                
                # 逆方向
                nx, ny = x - d*dx, y - d*dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if state[ny][nx] == player:  # 自分の石
                        score += 1.0 / (d + 1)
                    elif state[ny][nx] == -player:  # 相手の石
                        score += 0.5 / (d + 1)
        
        return score