import numpy as np
from GomokuEnv import GomokuEnv
from mcts import MCTSPlayer
from game import Game
import torch
from collections import deque
import random
import torch.nn.functional as F
from network import PolicyValueNet
import os
import matplotlib
matplotlib.use('Agg')  # バックエンドを非インタラクティブなものに変更
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import time
import pickle

BOARD_SIZE = 8  # ボードサイズ
N_IN_ROW = 5   # 勝利条件（連続する石の数）

def parallel_data_augmentation(args):
    """データ拡張を並列で実行するための関数"""
    state, mcts_prob, winner, board_size = args
    extend_data = []
    
    # stateをNumPy配列に変換し、4次元形状に変換
    state = np.array(state)
    mcts_prob = np.array(mcts_prob)
    
    # stateが1次元の場合、4次元形状に変換
    if len(state.shape) == 1:
        state = state.reshape(4, board_size, board_size)
    elif len(state.shape) == 2:
        temp_state = np.zeros((4, board_size, board_size))
        temp_state[0] = state
        state = temp_state
    
    # mcts_probを2次元に変換
    mcts_prob_2d = mcts_prob.reshape(board_size, board_size)
    
    # 元のデータを追加
    extend_data.append((state.flatten(), mcts_prob_2d.flatten(), winner))
    
    # 回転による拡張（90度、180度、270度）
    for i in [1, 2, 3]:
        # 各チャンネルを回転
        equi_state = np.array([np.rot90(state[j], i) for j in range(4)])
        # 確率分布も同じように回転
        equi_mcts_prob = np.rot90(mcts_prob_2d, i)
        extend_data.append((equi_state.flatten(),
                            equi_mcts_prob.flatten(),
                            winner))
                            
        # 水平反転
        equi_state_flip = np.array([np.fliplr(equi_state[j]) for j in range(4)])
        equi_mcts_prob_flip = np.fliplr(equi_mcts_prob)
        extend_data.append((equi_state_flip.flatten(),
                            equi_mcts_prob_flip.flatten(),
                            winner))
    
    # 元の状態の水平反転のみ
    equi_state_flip = np.array([np.fliplr(state[j]) for j in range(4)])
    equi_mcts_prob_flip = np.fliplr(mcts_prob_2d)
    extend_data.append((equi_state_flip.flatten(),
                        equi_mcts_prob_flip.flatten(),
                        winner))
    
    return extend_data

def parallel_selfplay_worker(args):
    """
    並列自己対戦を実行するワーカー関数
    各プロセスで独立して自己対戦を実行し、結果を返す
    """
    worker_id, model_state_dict, board_size, c_puct, n_playout, temp, game_count = args
    
    try:
        # 各プロセスでモデルを初期化
        env = GomokuEnv(board_size=board_size)
        policy_value_net = PolicyValueNet(board_size, env=env)
        
        # モデルの重みをロード
        policy_value_net.policy_value_net.load_state_dict(model_state_dict)
        policy_value_net.policy_value_net.eval()
        
        # MCTSプレイヤーを作成
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=c_puct,
                                 n_playout=n_playout,
                                 is_selfplay=True)
        
        # ゲームオブジェクトを作成
        game = Game(env, board_size=board_size)
        
        # 自己対戦データを収集
        all_play_data = []
        
        for i in range(game_count):
            try:
                winner, play_data = game.start_self_play(mcts_player, temp=temp)
                play_data = list(play_data)
                
                # データを収集
                if len(play_data) > 0:
                    all_play_data.extend(play_data)
                    
            except Exception as e:
                print(f"ワーカー {worker_id} でゲーム {i} 中にエラー: {e}")
                continue
        
        print(f"ワーカー {worker_id}: {game_count} ゲーム完了, {len(all_play_data)} データ収集")
        return all_play_data
        
    except Exception as e:
        print(f"ワーカー {worker_id} でエラー: {e}")
        return []

class AlphaZeroParallel:
    def __init__(self, parallel_game_count=None):
        self.board_size = BOARD_SIZE
        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.game = Game(self.env, board_size=BOARD_SIZE)
        self.n_in_row = N_IN_ROW
        
        # トレーニング用パラメータ
        self.learn_rate = 2e-3  # 学習率
        self.lr_multiplier = 1.0  # 学習率の乗数、KLに基づいて調整
        self.temp = 1.0  # 温度パラメータ
        self.n_playout = 400     # 各着手ごとのプレイアウト回数
        self.c_puct = 5  # UCBスコアの探索項の係数
        self.buffer_size = 25000  # 経験再生バッファのサイズ（並列処理に対応して増量）
        self.batch_size = 512  # トレーニング時のバッチサイズ
        self.date_buffer = deque(maxlen=self.buffer_size)  # 経験再生バッファ
        self.epochs = 20  # 各更新ステップでのエポック数（基本値）
        self.min_epochs = 10  # 最小エポック数
        self.max_epochs = 30  # 最大エポック数
        self.kl_targ = 0.02  # KLダイバージェンスの目標値
        self.check_freq = 100  # モデル評価の頻度（100ゲームごと）
        self.game_batch_num = 5000  # 1回のトレーニングサイクルでプレイするゲーム数
        self.best_win_ratio = 0.0  # 最善モデルの勝率
        
        # 並列処理用パラメータ
        self.num_cpu_cores = multiprocessing.cpu_count()
        # デフォルトの並列ゲーム数はCPUコア数の2/3程度に設定（効率的な並列化）
        if parallel_game_count is None:
            self.parallel_game_count = max(2, min(self.num_cpu_cores - 1, 8))
        else:
            self.parallel_game_count = parallel_game_count
        
        # ワーカー数を並列ゲーム数に合わせて最適化（無駄なワーカーを作らない）
        self.num_workers = min(self.parallel_game_count, max(1, self.num_cpu_cores - 1))
        self.games_per_worker = max(1, self.parallel_game_count // self.num_workers)  # 各ワーカーが実行するゲーム数
        
        # バッファサイズの動的調整
        self._adjust_buffer_size()
        
        # Loss記録用
        self.loss_history = []  # Loss値の履歴
        self.entropy_history = []  # エントロピーの履歴
        self.epoch_usage_history = []  # 実際に使用されたエポック数の履歴
        
        # モデルの初期化
        self.policy_value_net = PolicyValueNet(self.board_size, env=self.env)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=True)
        
        # GPU設定
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.cuda.empty_cache()
            self.gpu_batch_size = 32
            print(f"GPU使用率を80%に制限しました")
        else:
            self.gpu_batch_size = 64
            
        print(f"並列処理設定:")
        print(f"  CPUコア数: {self.num_cpu_cores}")
        print(f"  並列ゲーム数: {self.parallel_game_count}")
        print(f"  ワーカー数: {self.num_workers}")
        print(f"  ゲーム/ワーカー: {self.games_per_worker}")
        print(f"  効率性: {self.parallel_game_count / self.num_workers:.2f} ゲーム/ワーカー")
        print(f"エポック設定: 基本={self.epochs}, 範囲={self.min_epochs}-{self.max_epochs} (適応的調整)")
        print(f"データ使用効率: {(self.batch_size * self.epochs) / self.buffer_size:.1%} (1回の更新あたり)")

    def get_equi_data(self, play_data):
        """
        収集したセルフプレイデータを回転や反転によって拡張します。
        CPUで並列実行します。
        """
        # 並列処理のためのタスクを準備
        tasks = []
        for state, mcts_prob, winner in play_data:
            tasks.append((state, mcts_prob, winner, self.board_size))
        
        # CPUで並列実行
        extend_data = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(parallel_data_augmentation, task) for task in tasks]
            
            for future in as_completed(futures):
                extend_data.extend(future.result())
        
        return extend_data

    def collect_selfplay_data_parallel(self, total_games):
        """
        並列処理により複数のプロセスで自己対戦データを収集します。
        効率的なワーカー割り当てにより、各ワーカーが複数のゲームを実行します。
        """
        print(f"並列自己対戦開始: {total_games} ゲーム, {self.num_workers} ワーカー")
        
        # 現在のモデルの状態を取得
        model_state_dict = self.policy_value_net.policy_value_net.state_dict()
        
        # 効率的なワーカー割り当て
        # ワーカー数が並列ゲーム数を超えないように調整
        effective_workers = min(self.num_workers, total_games)
        base_games_per_worker = total_games // effective_workers
        remaining_games = total_games % effective_workers
        
        # ワーカー用のタスクを準備
        tasks = []
        for i in range(effective_workers):
            worker_games = base_games_per_worker + (1 if i < remaining_games else 0)
            if worker_games > 0:
                tasks.append((i, model_state_dict, self.board_size, self.c_puct, 
                              self.n_playout, self.temp, worker_games))
        
        print(f"ワーカー割り当て: {effective_workers} ワーカー, 平均 {base_games_per_worker} ゲーム/ワーカー")
        
        # ProcessPoolExecutorを使用して並列実行
        all_play_data = []
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(parallel_selfplay_worker, task) for task in tasks]
            
            for future in as_completed(futures):
                try:
                    worker_data = future.result()
                    all_play_data.extend(worker_data)
                except Exception as e:
                    print(f"ワーカーでエラー: {e}")
        
        end_time = time.time()
        print(f"並列自己対戦完了: {len(all_play_data)} データ収集, "
              f"実行時間: {end_time - start_time:.2f}秒")
        
        # データ拡張を実行
        if len(all_play_data) > 0:
            print("データ拡張を実行中...")
            augmented_data = self.get_equi_data(all_play_data)
            self.date_buffer.extend(augmented_data)
            print(f"データ拡張完了: {len(augmented_data)} データ")
        
        return len(all_play_data)

    def policy_update(self):
        """
        収集したデータからミニバッチを作成し、ポリシーとバリューネットワークを更新します。
        """
        # 経験再生バッファからミニバッチをサンプリング
        mini_batch = random.sample(self.date_buffer, self.batch_size)
        
        # CPUで並列処理でデータを準備
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # バッチデータを並列で処理
            futures = []
            batch_size_per_thread = len(mini_batch) // self.num_workers
            
            for i in range(self.num_workers):
                start_idx = i * batch_size_per_thread
                if i == self.num_workers - 1:
                    end_idx = len(mini_batch)
                else:
                    end_idx = (i + 1) * batch_size_per_thread
                
                thread_batch = mini_batch[start_idx:end_idx]
                futures.append(executor.submit(self._prepare_batch_data, thread_batch))
            
            # 結果を結合
            all_states = []
            all_mcts_probs = []
            all_winners = []
            
            for future in as_completed(futures):
                states, mcts_probs, winners = future.result()
                all_states.extend(states)
                all_mcts_probs.extend(mcts_probs)
                all_winners.extend(winners)
        
        # NumPy配列に変換
        state_batch = np.array(all_states)
        mcts_probs_batch = np.array(all_mcts_probs)
        winner_batch = np.array(all_winners)
        
        # 状態バッチを4次元形状に変換
        if len(state_batch.shape) == 2:
            state_batch = state_batch.reshape(-1, 4, self.board_size, self.board_size)
        
        # 更新前のポリシーとバリューを取得
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        # 適応的エポック数の決定
        adaptive_epochs = self._get_adaptive_epochs()
        
        # エポック数だけトレーニングを繰り返す
        for i in range(adaptive_epochs):
            # ネットワークを1ステップ学習させる
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            
            # 更新後のポリシーとバリューを取得
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            
            # 更新前後のポリシーのKLダイバージェンスを計算
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1))
            
            # KLが目標値の4倍を超えたら、学習を早期終了
            if kl > self.kl_targ * 4:
                print(f"早期終了: エポック {i+1}/{adaptive_epochs}, KL={kl:.5f}")
                break
            
            # KLダイバージェンスに基づいて学習率を調整
            if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
            elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
            
            # 学習の進捗を評価するための指標を計算
            winner_var = np.var(np.array(winner_batch))
            if winner_var > 1e-8:
                explained_var_old = (1 -
                                     np.var(np.array(winner_batch) - old_v.flatten()) /
                                     winner_var)
                explained_var_new = (1 -
                                     np.var(np.array(winner_batch) - new_v.flatten()) /
                                     winner_var)
            else:
                explained_var_old = 0.0
                explained_var_new = 0.0
            
            # 学習状況を出力
            if i % 1 == 0 or i == adaptive_epochs - 1:  # 1エポックごとまたは最後に出力
                print(("エポック {}/{}: kl:{:.5f}, "
                       "lr_multiplier:{:.3f}, "
                       "loss:{:.4f}, "
                       "entropy:{:.4f}, "
                       "explained_var_old:{:.3f}, "
                       "explained_var_new:{:.3f}"
                       ).format(i+1, adaptive_epochs, kl,
                                self.lr_multiplier,
                                loss,
                                entropy,
                                explained_var_old,
                                explained_var_new))
        
        # 最終的な学習状況を出力
        print(f"トレーニング完了: {i+1}/{adaptive_epochs} エポック実行")
        
        # Loss値を記録
        self.loss_history.append(loss)
        self.entropy_history.append(entropy)
        self.epoch_usage_history.append(i+1)  # 実際に使用されたエポック数を記録
        
        return loss, entropy

    def _get_adaptive_epochs(self):
        """
        現在の学習状況に基づいて適応的なエポック数を決定します。
        """
        # バッファサイズに基づく調整
        buffer_ratio = len(self.date_buffer) / self.buffer_size
        
        # 学習履歴に基づく調整
        if len(self.loss_history) > 10:
            # 最近10回の損失の変化率を計算
            recent_losses = self.loss_history[-10:]
            loss_trend = (recent_losses[-1] - recent_losses[0]) / recent_losses[0]
            
            # 損失が減少傾向の場合は多めのエポック、増加傾向の場合は少なめ
            if loss_trend < -0.1:  # 損失が10%以上減少
                epoch_adjustment = 5
            elif loss_trend > 0.1:  # 損失が10%以上増加
                epoch_adjustment = -5
            else:
                epoch_adjustment = 0
        else:
            epoch_adjustment = 0
        
        # バッファの充実度に基づく調整
        if buffer_ratio < 0.5:
            buffer_adjustment = -2  # バッファが少ない場合は少なめ
        elif buffer_ratio > 0.8:
            buffer_adjustment = 3   # バッファが充実している場合は多め
        else:
            buffer_adjustment = 0
        
        # 最終的なエポック数を計算
        adaptive_epochs = self.epochs + epoch_adjustment + buffer_adjustment
        
        # 最小・最大値で制限
        adaptive_epochs = max(self.min_epochs, min(self.max_epochs, adaptive_epochs))
        
        return adaptive_epochs

    def _prepare_batch_data(self, batch):
        """バッチデータを準備する（CPUで実行）"""
        states = [data[0] for data in batch]
        mcts_probs = [data[1] for data in batch]
        winners = [data[2] for data in batch]
        return states, mcts_probs, winners

    def save_loss_graph(self):
        """
        Loss値のグラフを保存します。
        """
        if len(self.loss_history) > 0:
            plt.figure(figsize=(12, 5))
            
            # Loss値のグラフ
            plt.subplot(1, 2, 1)
            plt.plot(self.loss_history, 'b-', label='Loss')
            plt.xlabel('Update Step')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)
            
            # エントロピーのグラフ
            plt.subplot(1, 2, 2)
            plt.plot(self.entropy_history, 'r-', label='Entropy')
            plt.xlabel('Update Step')
            plt.ylabel('Entropy')
            plt.title('Policy Entropy')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('./loss_graph.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Loss グラフを './loss_graph.png' に保存しました。")

    def train(self):
        """
        AlphaZeroの並列トレーニングパイプラインを実行します。
        自己対戦データの収集、ポリシーの更新、モデルの評価を繰り返します。
        """
        try:
            # トレーニング開始時に初期の最善モデルを保存
            if not os.path.exists('./best_policy.model'):
                print("初期の最善ポリシーモデルを保存しています...")
                self.policy_value_net.save_model('./best_policy.model')
                print("初期の最善ポリシーモデルを保存しました。")
            
            # 指定されたゲームバッチ数だけトレーニングサイクルを繰り返す
            for i in range(self.game_batch_num):
                cycle_start_time = time.time()
                print(f"\n=== トレーニングサイクル {i+1}/{self.game_batch_num} ===")
                
                # 並列自己対戦データを収集
                collected_games = self.collect_selfplay_data_parallel(self.parallel_game_count)
                print(f"サイクル {i+1}: {collected_games} ゲーム完了")
                
                # バッファに十分なデータが溜まったらポリシーを更新
                if len(self.date_buffer) >= self.batch_size:
                    print("ポリシーを更新中...")
                    loss, entropy = self.policy_update()
                    self.save_loss_graph()
                    print(f"ポリシー更新完了: Loss={loss:.4f}, Entropy={entropy:.4f}")
                
                # 一定の頻度で現在のモデルを評価
                if (i + 1) % self.check_freq == 0:
                    print(f"\n=== モデル評価 (サイクル {i+1}) ===")
                    win_ratio = self.policy_evaluate()
                    
                    # 現在のポリシーを保存
                    self.policy_value_net.save_model('./current_policy.model')
                    
                    # Lossグラフを保存
                    self.save_loss_graph()
                    
                    # 新しいモデルが最善モデルを上回った場合
                    if win_ratio > max(self.best_win_ratio, 0.51):
                        print(f"新しい最善ポリシーが見つかりました！（勝率: {win_ratio:.3f}）")
                        self.best_win_ratio = win_ratio
                        # 最善ポリシーを更新して保存
                        self.policy_value_net.save_model('./best_policy.model')
                
                # サイクルの実行時間を計算・表示
                cycle_end_time = time.time()
                cycle_duration = cycle_end_time - cycle_start_time
                print(f"サイクル {i+1} 実行時間: {cycle_duration:.2f}秒")
                        
        except KeyboardInterrupt:
            print('\n\rトレーニングを中断しました。')
            # 最終的なグラフを保存
            self.save_loss_graph()
            # 現在のモデルを保存
            self.policy_value_net.save_model('./interrupted_policy.model')
            print("中断時のモデルを保存しました。")

    def policy_evaluate(self, n_games=10):
        """
        現在のポリシーと最善のポリシーを比較評価します。
        n_games回対戦し、現在のポリシーの勝率を計算します。
        """
        # 現在のポリシーを持つMCTSプレイヤーを作成
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout + 20,
                                         is_selfplay=False)
        
        # 最善のポリシーをロードしてMCTSプレイヤーを作成
        best_policy = PolicyValueNet(self.board_size, env=self.env)
        try:
            if os.path.exists('./best_policy.model'):
                best_policy.load_model('./best_policy.model')
                print("最善のポリシーモデルをロードしました。")
            else:
                print("最善のポリシーモデルが見つかりません。現在のモデルを最善として保存します。")
                self.policy_value_net.save_model('./best_policy.model')
                return 0.55
        except Exception as e:
            print(f"最善のポリシーモデルのロードに失敗しました: {e}")
            self.policy_value_net.save_model('./best_policy.model')
            return 0.55

        best_mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=False)

        win_cnt = 0
        draw_cnt = 0
        current_wins_as_first = 0
        current_wins_as_second = 0
        
        print("評価開始: 現在のモデル vs 最善のモデル")
        
        # n_games/2 回、現在のプレイヤーが先手で対戦
        for i in range(n_games // 2):
            winner = self.game.start_play(current_mcts_player,
                                          best_mcts_player,
                                          start_player=0,
                                          is_shown=0)
            if winner == 1:
                win_cnt += 1
                current_wins_as_first += 1
            elif winner == 0:
                draw_cnt += 1
            print(f"先手戦 {i+1}/{n_games//2}: 勝者 = {winner}")
        
        # n_games/2 回、現在のプレイヤーが後手で対戦
        for i in range(n_games // 2):
            winner = self.game.start_play(best_mcts_player,
                                          current_mcts_player,
                                          start_player=0,
                                          is_shown=0)
            if winner == -1:
                win_cnt += 1
                current_wins_as_second += 1
            elif winner == 0:
                draw_cnt += 1
            print(f"後手戦 {i+1}/{n_games//2}: 勝者 = {winner}")
        
        win_ratio = win_cnt / n_games
        print(f"詳細結果:")
        print(f"  先手での勝利: {current_wins_as_first}/{n_games//2}")
        print(f"  後手での勝利: {current_wins_as_second}/{n_games//2}")
        print(f"  引き分け: {draw_cnt}/{n_games}")
        print(f"  総合結果: {win_cnt}勝 / {n_games}戦, 勝率: {win_ratio:.3f}")
        return win_ratio

    def _adjust_buffer_size(self):
        """
        並列処理の設定に基づいて適切なバッファサイズを動的に調整します。
        """
        # 1ゲームあたりの平均手数を推定（8x8ボードで約20-40手）
        estimated_moves_per_game = 30
        
        # データ拡張による倍率（回転4回 + 反転4回 = 8倍）
        data_augmentation_factor = 8
        
        # 1バッチあたりの推定データ数
        data_per_batch = (self.parallel_game_count * 
                         estimated_moves_per_game * 
                         data_augmentation_factor)
        
        # 推奨バッファサイズ（20-30バッチ分のデータを保持）
        recommended_buffer_size = data_per_batch * 25
        
        # 最小バッファサイズ（バッチサイズの20倍以上）
        min_buffer_size = self.batch_size * 20
        
        # 最終的なバッファサイズを決定
        adjusted_buffer_size = max(recommended_buffer_size, min_buffer_size)
        
        # 元の設定値と比較して調整
        if adjusted_buffer_size != self.buffer_size:
            print(f"バッファサイズを調整: {self.buffer_size} → {adjusted_buffer_size}")
            self.buffer_size = adjusted_buffer_size
            # dequeを再作成
            if hasattr(self, 'date_buffer'):
                old_data = list(self.date_buffer)
                self.date_buffer = deque(old_data, maxlen=self.buffer_size)
            else:
                self.date_buffer = deque(maxlen=self.buffer_size)
        
        print(f"推定データ量: {data_per_batch} データ/バッチ")
        print(f"バッファサイズ: {self.buffer_size} （約{self.buffer_size // data_per_batch}バッチ分）")

if __name__ == "__main__":
    # マルチプロセシングのスタートメソッドを設定
    multiprocessing.set_start_method('spawn', force=True)
    
    # 並列AlphaZeroトレーニングを開始
    # 並列ゲーム数を指定可能（デフォルトは自動調整）
    trainer = AlphaZeroParallel(parallel_game_count=None)  # Noneで自動調整、数値で手動指定
    trainer.train()
