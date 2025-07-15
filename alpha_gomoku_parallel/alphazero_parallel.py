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
import queue
import copy
from torch.multiprocessing import Pool, set_start_method
import warnings
warnings.filterwarnings('ignore')

BOARD_SIZE = 8  # ボードサイズ
N_IN_ROW = 5 # 勝利条件（連続する石の数）

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

def parallel_selfplay_game(args):
    """並列で自己対戦を実行するための関数"""
    network_weights, board_size, n_playout, c_puct, temp = args
    
    try:
        # 各プロセスで独立したネットワークを作成
        env = GomokuEnv(board_size=board_size)
        policy_value_net = PolicyValueNet(board_size, env=env, use_gpu=False)  # プロセス間ではCPUを使用
        
        # ネットワークの重みを設定
        policy_value_net.policy_value_net.load_state_dict(network_weights)
        
        game = Game(env, board_size=board_size)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                               c_puct=c_puct, n_playout=n_playout,
                               is_selfplay=True)
        
        # 自己対戦を実行
        winner, play_data = game.start_self_play(mcts_player, temp=temp)
        
        # メモリ効率を改善するため、必要最小限のデータのみを返す
        return winner, list(play_data)
        
    except Exception as e:
        print(f"プロセス内でエラーが発生: {e}")
        return None, []

class AlphaZeroParallel:
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.game = Game(self.env, board_size=BOARD_SIZE)
        self.n_in_row = N_IN_ROW
        
        # ハードウェア情報を取得
        self.num_cpu_cores = multiprocessing.cpu_count()
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        
        print(f"システム情報:")
        print(f"  CPU cores: {self.num_cpu_cores}")
        print(f"  GPU available: {self.gpu_available}")
        print(f"  GPU count: {self.gpu_count}")
        
        # トレーニング用パラメータ
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        
        # バッファサイズの動的調整（ボードサイズに基づく）
        # 64GBメモリを活用した大容量バッファ設定
        if self.board_size <= 8:
            # 8x8: 100,000 - 200,000 サンプル（約2-4GB）
            base_buffer_size = 150000
            max_memory_mb = 4000  # 4GB
        elif self.board_size <= 15:
            # 15x15: 80,000 - 120,000 サンプル（約3-5GB）
            base_buffer_size = 100000
            max_memory_mb = 6000  # 6GB
        else:
            # 19x19: 60,000 - 80,000 サンプル（約4-6GB）
            base_buffer_size = 70000
            max_memory_mb = 8000  # 8GB
        
        self.buffer_size = base_buffer_size
        
        # メモリ上限チェック（念のため）
        data_size_per_point = 10.4  # KB（データ拡張込み）
        max_buffer_size = int(max_memory_mb * 1024 / data_size_per_point)
        self.buffer_size = min(self.buffer_size, max_buffer_size)
        
        # 大容量メモリを活用したバッチサイズ調整
        if self.gpu_available:
            # GPU使用時：より大きなバッチサイズで効率的な学習
            self.batch_size = 1024  # 従来の512から増加
            self.training_batch_size = 128  # GPU使用時
            self.gpu_batch_size = 64
        else:
            # CPU使用時：メモリを最大限活用
            self.batch_size = 768  # 従来の512から増加
            self.training_batch_size = 64  # CPU使用時
            self.gpu_batch_size = 32
        
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 20
        self.kl_targ = 0.02
        self.check_freq = 100
        self.game_batch_num = 5000
        self.best_win_ratio = 0.0
        
        # 並列処理の設定（大容量メモリを活用）
        # CPU並列でのゲーム数を増加
        self.parallel_games = min(self.num_cpu_cores - 1, 16)  # 最大16並列
        self.data_augmentation_workers = min(self.num_cpu_cores // 2, 8)  # 最大8ワーカー
        
        # Loss記録用
        self.loss_history = []
        self.entropy_history = []
        
        # 学習とゲーム実行の同期用
        self.game_data_queue = queue.Queue()
        self.learning_thread = None
        self.learning_active = True
        self.learning_lock = threading.Lock()
        
        # 並列処理用のプロセスプールを事前に初期化
        self.game_process_pool = ProcessPoolExecutor(max_workers=self.parallel_games)
        
        # モデルの初期化
        use_gpu = torch.cuda.is_available()
        self.policy_value_net = PolicyValueNet(self.board_size, env=self.env, use_gpu=use_gpu)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                     c_puct=self.c_puct, n_playout=self.n_playout,
                                     is_selfplay=True)
        
        # GPU使用率を制限
        if self.gpu_available:
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.cuda.empty_cache()
            print(f"GPU使用率を80%に制限しました")
        
        print(f"並列処理設定:")
        print(f"  並列ゲーム数: {self.parallel_games}")
        print(f"  データ拡張ワーカー数: {self.data_augmentation_workers}")
        print(f"  学習バッチサイズ: {self.training_batch_size}")
        print(f"  リプレイバッファサイズ: {self.buffer_size:,}")
        print(f"  推定メモリ使用量: {self.buffer_size * 10.4 / 1024:.1f} MB")
    
    def get_equi_data(self, play_data):
        """
        データ拡張を並列で実行
        """
        tasks = []
        for state, mcts_prob, winner in play_data:
            tasks.append((state, mcts_prob, winner, self.board_size))
        
        extend_data = []
        with ThreadPoolExecutor(max_workers=self.data_augmentation_workers) as executor:
            futures = [executor.submit(parallel_data_augmentation, task) for task in tasks]
            
            for future in as_completed(futures):
                extend_data.extend(future.result())
        
        return extend_data
    
    def collect_parallel_selfplay_data(self, num_games):
        """
        並列で自己対戦データを収集
        """
        if num_games <= 1:
            # 1ゲームの場合は並列処理のオーバーヘッドを避ける
            return self._collect_single_game()
        
        # 現在のネットワークの重みを取得
        network_weights = self.policy_value_net.policy_value_net.state_dict()
        
        # 並列実行のためのタスクを準備
        tasks = []
        for _ in range(num_games):
            tasks.append((network_weights, self.board_size, self.n_playout, 
                         self.c_puct, self.temp))
        
        all_play_data = []
        
        # 事前に初期化されたプロセスプールを使用
        futures = [self.game_process_pool.submit(parallel_selfplay_game, task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                winner, play_data = future.result()
                # データ拡張を即座に実行
                extended_data = self.get_equi_data(play_data)
                all_play_data.extend(extended_data)
                
                # 学習キューにデータを追加
                self.game_data_queue.put(extended_data)
                
            except Exception as e:
                print(f"並列ゲーム実行中のエラー: {e}")
        
        return all_play_data
    
    def _collect_single_game(self):
        """
        単一ゲームの効率的な実行
        """
        winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
        play_data = list(play_data)
        extended_data = self.get_equi_data(play_data)
        self.game_data_queue.put(extended_data)
        return extended_data
    
    def continuous_learning_thread(self):
        """
        継続的な学習スレッド
        ゲームデータが利用可能になったら即座に学習を開始
        """
        accumulated_data = []
        
        while self.learning_active:
            try:
                # タイムアウト付きでデータを待機
                game_data = self.game_data_queue.get(timeout=1.0)
                accumulated_data.extend(game_data)
                
                # データバッファに追加
                self.data_buffer.extend(game_data)
                
                # 十分なデータが溜まったら学習実行
                if len(self.data_buffer) >= self.batch_size:
                    with self.learning_lock:
                        loss, entropy = self.policy_update()
                        self.loss_history.append(loss)
                        self.entropy_history.append(entropy)
                        
                        print(f"学習完了 - Loss: {loss:.4f}, Entropy: {entropy:.4f}, "
                              f"Buffer size: {len(self.data_buffer)}")
                
                # キューのタスク完了をマーク
                self.game_data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"学習スレッドエラー: {e}")
                continue
    
    def policy_update(self):
        """
        ポリシー更新（GPU並列処理対応）
        """
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        
        # データ準備を並列で実行
        with ThreadPoolExecutor(max_workers=self.data_augmentation_workers) as executor:
            futures = []
            batch_size_per_thread = len(mini_batch) // self.data_augmentation_workers
            
            for i in range(self.data_augmentation_workers):
                start_idx = i * batch_size_per_thread
                if i == self.data_augmentation_workers - 1:
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
        
        # 更新前のポリシーとバリューを取得（NumPy配列のまま）
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        # エポック数だけトレーニングを繰り返す
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            
            # KLダイバージェンスを計算（CPUで実行）
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1))
            
            if kl > self.kl_targ * 4:
                break
            
            # 学習率調整
            if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
            elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
        
        return loss, entropy
    
    def _prepare_batch_data(self, batch):
        """バッチデータを準備"""
        states = [data[0] for data in batch]
        mcts_probs = [data[1] for data in batch]
        winners = [data[2] for data in batch]
        return states, mcts_probs, winners
    
    def save_loss_graph(self):
        """Loss値のグラフを保存"""
        if len(self.loss_history) > 0:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.loss_history, 'b-', label='Loss')
            plt.xlabel('Update Step')
            plt.ylabel('Loss')
            plt.title('Training Loss (Parallel)')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.entropy_history, 'r-', label='Entropy')
            plt.xlabel('Update Step')
            plt.ylabel('Entropy')
            plt.title('Policy Entropy (Parallel)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('./loss_graph_parallel.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("並列処理版Loss グラフを './loss_graph_parallel.png' に保存しました。")
    
    def train(self):
        """
        並列化されたAlphaZeroトレーニングパイプライン
        """
        try:
            # 学習スレッドを開始
            self.learning_thread = threading.Thread(target=self.continuous_learning_thread)
            self.learning_thread.daemon = True
            self.learning_thread.start()
            print("継続的学習スレッドを開始しました。")
            
            # 初期モデル保存
            if not os.path.exists('./best_policy_parallel.model'):
                print("初期の最善ポリシーモデルを保存しています...")
                self.policy_value_net.save_model('./best_policy_parallel.model')
                print("初期の最善ポリシーモデルを保存しました。")
            
            # トレーニングループ
            for i in range(self.game_batch_num):
                start_time = time.time()
                
                # 並列で自己対戦データを収集
                play_data = self.collect_parallel_selfplay_data(self.parallel_games)
                
                collection_time = time.time() - start_time
                
                print(f"バッチ {i+1}/{self.game_batch_num} 完了 - "
                      f"並列ゲーム: {self.parallel_games}, "
                      f"収集時間: {collection_time:.2f}秒, "
                      f"データ数: {len(play_data)}")
                
                # 一定の頻度で評価
                if (i + 1) % self.check_freq == 0:
                    # 学習の完了を待機
                    self.game_data_queue.join()
                    
                    print(f"現在の自己対戦バッチ: {i+1}")
                    win_ratio = self.policy_evaluate()
                    
                    # モデル保存
                    self.policy_value_net.save_model('./current_policy_parallel.model')
                    self.save_loss_graph()
                    
                    # 最善モデル更新
                    if win_ratio > max(self.best_win_ratio, 0.51):
                        print(f"新しい最善ポリシーが見つかりました！（勝率: {win_ratio:.3f}）")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model('./best_policy_parallel.model')
                        
        except KeyboardInterrupt:
            print('\n\rトレーニングを中断しました。')
        finally:
            # 学習スレッドを終了
            self.learning_active = False
            if self.learning_thread:
                self.learning_thread.join()
            
            # プロセスプールを終了
            if hasattr(self, 'game_process_pool'):
                self.game_process_pool.shutdown(wait=True)
                
            self.save_loss_graph()
            print("並列処理を終了しました。")
    
    def policy_evaluate(self, n_games=10):
        """
        ポリシー評価（並列処理版）
        """
        # 現在のモデルでMCTSプレイヤーを作成
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                        c_puct=self.c_puct,
                                        n_playout=self.n_playout + 20,
                                        is_selfplay=False)
        
        # 最善モデルをロード
        best_policy = PolicyValueNet(self.board_size, env=self.env, use_gpu=torch.cuda.is_available())
        try:
            if os.path.exists('./best_policy_parallel.model'):
                best_policy.load_model('./best_policy_parallel.model')
                print("最善のポリシーモデルをロードしました。")
            else:
                print("最善のポリシーモデルが見つかりません。現在のモデルを最善として保存します。")
                self.policy_value_net.save_model('./best_policy_parallel.model')
                return 0.55
        except Exception as e:
            print(f"最善のポリシーモデルのロードに失敗しました: {e}")
            self.policy_value_net.save_model('./best_policy_parallel.model')
            return 0.55
        
        best_mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                     c_puct=self.c_puct,
                                     n_playout=self.n_playout,
                                     is_selfplay=False)
        
        win_cnt = 0
        draw_cnt = 0
        
        print("評価開始: 現在のモデル vs 最善のモデル（並列処理版）")
        
        # 並列でゲームを実行
        game_results = []
        
        # 先手戦
        for i in range(n_games // 2):
            winner = self.game.start_play(current_mcts_player,
                                         best_mcts_player,
                                         start_player=0,
                                         is_shown=0)
            game_results.append(winner)
            if winner == 1:
                win_cnt += 1
            elif winner == 0:
                draw_cnt += 1
        
        # 後手戦
        for i in range(n_games // 2):
            winner = self.game.start_play(best_mcts_player,
                                         current_mcts_player,
                                         start_player=0,
                                         is_shown=0)
            game_results.append(winner)
            if winner == -1:
                win_cnt += 1
            elif winner == 0:
                draw_cnt += 1
        
        win_ratio = win_cnt / n_games
        print(f"並列処理版評価結果: {win_cnt}勝 / {n_games}戦, 勝率: {win_ratio:.3f}")
        return win_ratio

if __name__ == "__main__":
    # マルチプロセス用の設定
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    # 並列化されたAlphaZeroを実行
    alpha_zero = AlphaZeroParallel()
    alpha_zero.train()
