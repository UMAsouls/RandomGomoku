"""
MCTS実装
"""

import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import concurrent.futures
import threading
import time
from tqdm import tqdm

from .node import MCTSNode
from .game_utils import (
    check_win_pattern_numba, get_legal_moves, get_winner_value,
    detect_winning_move, detect_blocking_move, has_open_four,
    is_winning_move, is_terminal, augment_data
)
from ..config import (
    device, MCTS_NUM_SIMULATIONS, MCTS_C_PUCT, MCTS_USE_GUMBEL, 
    MCTS_GUMBEL_SCALE, MCTS_ADD_ROOT_NOISE, MCTS_DIRICHLET_ALPHA, 
    MCTS_DIRICHLET_WEIGHT, MCTS_TEMPERATURE_THRESHOLD, 
    MCTS_INITIAL_TEMPERATURE, MCTS_FINAL_TEMPERATURE
)


class MCTS:
    """モンテカルロ木探索の実装"""
    
    def __init__(self, model, num_simulations=MCTS_NUM_SIMULATIONS, c_puct=MCTS_C_PUCT, 
                 use_gumbel=MCTS_USE_GUMBEL, gumbel_scale=MCTS_GUMBEL_SCALE, 
                 add_root_noise=MCTS_ADD_ROOT_NOISE, dirichlet_alpha=MCTS_DIRICHLET_ALPHA, 
                 dirichlet_weight=MCTS_DIRICHLET_WEIGHT, temperature_threshold=MCTS_TEMPERATURE_THRESHOLD, 
                 initial_temperature=MCTS_INITIAL_TEMPERATURE, final_temperature=MCTS_FINAL_TEMPERATURE):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.board_size = model.board_size
        self.use_gumbel = use_gumbel
        self.gumbel_scale = gumbel_scale
        
        # ルートノイズ関連のパラメータ
        self.add_root_noise = add_root_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        
        # ボルツマン分布の温度パラメータ
        self.temperature_threshold = temperature_threshold
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        
        # スレッドロック
        self.lock = threading.Lock()
        
        # 統計情報
        self.stats = {
            'total_searches': 0,
            'total_simulations': 0,
            'winning_moves_found': 0,
            'blocking_moves_found': 0,
            'average_search_time': 0.0,
            'search_times': []
        }

    def search(self, state, last_move=None):
        """与えられた状態に基づいてMCTSを実行"""
        start_time = time.time()
        
        root = MCTSNode(0)
        root.state = state.copy()
        
        # 現在のプレイヤーを特定
        current_player = 1 if np.sum(state == 1) == np.sum(state == -1) else -1
        
        # 評価関数から方策と価値を取得
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value_output = self.model(state_tensor, last_move, current_player)
        
        # policy_logitsは生のlogitsなので、softmaxを適用して確率分布に変換
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        
        # value_outputは勝率（0から1の範囲）を直接出力
        value = value_output.squeeze(0).item()
        # 勝率を-1から1の範囲に変換
        value = 2.0 * value - 1.0
        
        policy_legal = np.zeros(self.board_size * self.board_size)
        legal_moves = get_legal_moves(state)
        
        # 勝利パターンと防御パターンを効率的に検出
        winning_move = detect_winning_move(state.copy(), current_player, self.board_size)
        if winning_move is not None:
            # 勝利確定手があれば即座に選択
            self.stats['winning_moves_found'] += 1
            policy_legal = np.zeros(self.board_size * self.board_size)
            policy_legal[winning_move] = 1.0
            self._update_search_stats(start_time)
            return policy_legal
            
        # 相手の勝利を防ぐ必要がある場合
        blocking_move = detect_blocking_move(state.copy(), current_player, self.board_size)
        if blocking_move is not None:
            # 防御が必要な場合は防御手を選択
            self.stats['blocking_moves_found'] += 1
            policy_legal = np.zeros(self.board_size * self.board_size)
            policy_legal[blocking_move] = 1.0
            self._update_search_stats(start_time)
            return policy_legal
        
        # 通常のMCTSで探索
        for move in legal_moves:
            policy_legal[move] = policy[move]
        
        # 合法手がある場合は正規化
        if len(legal_moves) > 0:
            policy_legal = policy_legal / np.sum(policy_legal)
        
        # ルートノードにディリクレノイズを適用
        if self.add_root_noise and len(legal_moves) > 0:
            dirichlet_noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            
            for i, move in enumerate(legal_moves):
                original_prob = policy_legal[move]
                noisy_prob = (1 - self.dirichlet_weight) * original_prob + self.dirichlet_weight * dirichlet_noise[i]
                policy_legal[move] = noisy_prob
                
            # 再正規化
            if np.sum(policy_legal) > 0:
                policy_legal = policy_legal / np.sum(policy_legal)
        
        # 子ノードの初期化
        for move in legal_moves:
            root.children[move] = MCTSNode(policy_legal[move])

        # 並列シミュレーション実行
        self._run_simulations_parallel(root, state.copy(), self.num_simulations)
        
        # 訪問回数から方策を計算
        visit_counts = np.zeros(self.board_size * self.board_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # 現在の手数と訪問回数分布に基づいて適応的な温度を決定
        temperature = self._get_adaptive_temperature(state, visit_counts)
        
        # ボルツマン分布を使用して方策を計算
        mcts_policy = self._boltzmann_policy(visit_counts, temperature)
        mcts_value = root.value()
        
        # 統計情報の更新
        self._update_search_stats(start_time)
        
        return mcts_policy, mcts_value

    def _run_simulations_parallel(self, root, state, num_simulations):
        """MCTSのシミュレーションを並列実行"""
        # シミュレーション数が少ない場合はシーケンシャル実行
        if num_simulations <= 32:
            for _ in range(num_simulations):
                self._run_single_simulation(root, state.copy())
            return
            
        # 並列実行のためのバッチ処理
        import multiprocessing as mp
        # CPUコア数に応じて最適化
        cpu_count = max(1, min(4, mp.cpu_count()))  # 最大4スレッドに制限
        batch_size = max(4, num_simulations // cpu_count)  # 最小バッチサイズを4に設定
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
            futures = []
            remaining_sims = num_simulations
            
            while remaining_sims > 0:
                current_batch = min(batch_size, remaining_sims)
                futures.append(executor.submit(self._run_batch_simulations, root, state.copy(), current_batch))
                remaining_sims -= current_batch
                
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
        last_move = None
        
        # 葉ノードを見つける
        while node.expanded():
            action, node = self._select_child(node)
            x, y = action % self.board_size, action // self.board_size
            last_move = (x, y)
            current_state[y][x] = -1 if np.sum(current_state == 1) > np.sum(current_state == -1) else 1
            search_path.append(node)
        
        # 葉ノードの状態を評価
        leaf_state = current_state
        
        # ターミナル状態かチェック
        terminal = is_terminal(leaf_state, self.board_size)
        value = 0
        
        if terminal:
            # ゲーム終了：勝者に基づいて価値を設定
            value = get_winner_value(leaf_state, self.board_size)
        else:
            # ノードを展開
            leaf_node = search_path[-1]
            leaf_node.state = leaf_state.copy()
            
            # ニューラルネットワークで評価
            leaf_tensor = torch.tensor(leaf_state, dtype=torch.float32, device=device).unsqueeze(0)
            current_player = 1 if np.sum(leaf_state == 1) == np.sum(leaf_state == -1) else -1
            with torch.no_grad():
                policy_logits, value_output = self.model(leaf_tensor, last_move, current_player)
            
            # policy_logitsは生のlogitsなので、softmaxを適用して確率分布に変換
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            
            # value_outputは勝率（0から1の範囲）を直接出力
            value = value_output.squeeze(0).item()
            # 勝率を-1から1の範囲に変換
            value = 2.0 * value - 1.0
            
            # 合法手の取得と方策の正規化
            legal_moves = get_legal_moves(leaf_state)
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
            node.update(value)
            value = -value  # 交互に手番が変わるので、価値を反転

    def _select_child(self, node):
        """UCBスコアに基づいて子ノードを選択"""
        best_score = float('-inf')
        best_action = -1
        best_child = None
        
        # スレッドセーフにするため、反復処理前に子ノードのリストをコピー
        with self.lock:
            items = list(node.children.items())
        random.shuffle(items)

        for action, child in items:
            score = self._ucb_score(node, child, action)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _ucb_score(self, parent, child, action):
        """UCB (Upper Confidence Bound) スコアの計算"""
        # 勝ち確定手なら無限大のスコア（早期チェック）
        if parent.state is not None:
            y, x = divmod(action, self.board_size)
            temp_state = parent.state.copy()
            player = 1 if np.sum(temp_state == 1) == np.sum(temp_state == -1) else -1
            temp_state[y][x] = player
            
            # 直接勝利する手は最優先
            if check_win_pattern_numba(temp_state, x, y, player, self.board_size):
                return float('inf')
        
        # 探索項：未訪問ノードを優先的に探索
        prior_score = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        
        # 価値項：平均価値を使用（-1から1の範囲）
        if child.visit_count > 0:
            value_score = child.value()
        else:
            value_score = 0
        
        return value_score + prior_score

    def _get_move_count(self, state):
        """盤面に置かれた石の数（着手数）を数える"""
        return np.count_nonzero(state)

    def _get_adaptive_temperature(self, state, visit_counts):
        """適応的な温度パラメータを計算"""
        move_count = self._get_move_count(state)
        
        # 基本温度：手数に基づく
        base_temperature = self.initial_temperature if move_count < self.temperature_threshold else self.final_temperature
        
        # 訪問回数の分散に基づく調整
        if np.sum(visit_counts) > 0:
            normalized_counts = visit_counts / np.sum(visit_counts)
            # エントロピーを計算（選択の多様性を測定）
            entropy = -np.sum(normalized_counts * np.log(normalized_counts + 1e-10))
            max_entropy = np.log(len(np.nonzero(visit_counts)[0]))
            
            # エントロピーが高い場合（選択が分散）は温度を上げる
            if max_entropy > 0:
                entropy_factor = 1.0 + 0.5 * (entropy / max_entropy)
                return base_temperature * entropy_factor
        
        return base_temperature

    def _boltzmann_policy(self, visit_counts, temperature):
        """ボルツマン分布を使用して訪問回数から方策を計算"""
        # 温度がほぼゼロの場合はグリーディー選択
        if temperature < 1e-6:
            max_indices = np.where(visit_counts == np.max(visit_counts))[0]
            action = np.random.choice(max_indices)  # 同点の場合はランダム選択
            policy = np.zeros(len(visit_counts))
            policy[action] = 1.0
            return policy
        
        # 訪問回数がゼロでないアクションのみを考慮
        nonzero_indices = np.where(visit_counts > 0)[0]
        if len(nonzero_indices) == 0:
            # すべての訪問回数がゼロの場合は一様分布
            return np.ones(len(visit_counts)) / len(visit_counts)
        
        if len(nonzero_indices) == 1:
            # 一つの選択肢のみの場合
            policy = np.zeros(len(visit_counts))
            policy[nonzero_indices[0]] = 1.0
            return policy
        
        try:
            # ボルツマン分布: P(a) ∝ exp(N(a)/τ)
            counts = visit_counts[nonzero_indices]
            
            # 数値安定性のために対数を使用してソフトマックスを計算
            logits = counts / temperature
            
            # 数値安定性のために最大値を引く
            max_logit = np.max(logits)
            shifted_logits = logits - max_logit
            
            # オーバーフロー防止
            shifted_logits = np.clip(shifted_logits, -700, 700)
            
            exp_logits = np.exp(shifted_logits)
            sum_exp = np.sum(exp_logits)
            
            # ゼロ除算防止
            if sum_exp == 0 or not np.isfinite(sum_exp):
                # フォールバック：最頻値を選択
                max_indices = np.where(counts == np.max(counts))[0]
                selected_idx = np.random.choice(max_indices)
                policy = np.zeros(len(visit_counts))
                policy[nonzero_indices[selected_idx]] = 1.0
                return policy
            
            # 正規化してボルツマン確率を計算
            boltzmann_probs = exp_logits / sum_exp
            
            # 方策の初期化
            policy = np.zeros(len(visit_counts))
            policy[nonzero_indices] = boltzmann_probs
            
            # 最終的な正規化チェック
            total_prob = np.sum(policy)
            if total_prob > 0:
                policy = policy / total_prob
                
            return policy
            
        except Exception as e:
            # エラーが発生した場合は最も訪問回数の多い行動を選択
            print(f"ボルツマン分布計算中にエラーが発生: {e}")
            max_indices = np.where(visit_counts == np.max(visit_counts))[0]
            action = np.random.choice(max_indices)
            policy = np.zeros(len(visit_counts))
            policy[action] = 1.0
            return policy

    def _update_search_stats(self, start_time):
        """検索統計の更新"""
        elapsed_time = time.time() - start_time
        self.stats['total_searches'] += 1
        self.stats['total_simulations'] += self.num_simulations
        self.stats['search_times'].append(elapsed_time)
        
        # 移動平均を計算（最新50回の平均）
        recent_times = self.stats['search_times'][-50:]
        self.stats['average_search_time'] = np.mean(recent_times)
    
    def get_stats(self):
        """統計情報を取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計情報をリセット"""
        self.stats = {
            'total_searches': 0,
            'total_simulations': 0,
            'winning_moves_found': 0,
            'blocking_moves_found': 0,
            'average_search_time': 0.0,
            'search_times': []
        }
        
