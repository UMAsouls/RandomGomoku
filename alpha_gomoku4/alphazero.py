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
from mcts import MCTSPlayer
from game import Game
import torch
from collections import deque
import random
import torch.nn.functional as F
from network import PolicyValueNet
from mcts import MCTSPlayer
import random
import matplotlib.pyplot as plt
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading

BOARD_SIZE = 6  # ボードサイズ
N_IN_ROW = 4 # 勝利条件（連続する石の数）

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

class AlphaZero:
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.game = Game(self.env, board_size=BOARD_SIZE)
        self.n_in_row = N_IN_ROW
        # トレーニング用パラメータ
        self.learn_rate = 2e-3  # 学習率
        self.lr_multiplier = 1.0  # 学習率の乗数、KLに基づいて調整
        self.temp = 1.0  # 温度パラメータ
        self.n_playout = 400     # 各着手ごとのプレイアウト回数
        self.c_puct = 4  # UCBスコアの探索項の係数
        self.buffer_size = 10000  # 経験再生バッファのサイズ
        self.batch_size = 512  # トレーニング時のバッチサイズ
        self.date_buffer = deque(maxlen=self.buffer_size)  # 経験再生バッファ
        self.play_batch_size = 8  # 自己対戦の並列実行数
        self.epochs =20  # 各更新ステップでのエポック数
        self.kl_targ = 0.02 # KLダイバージェンスの目標値
        self.check_freq = 50 # モデル評価の頻度（100ゲームごと）
        self.game_batch_num = 1500  # 1回のトレーニングサイクルでプレイするゲーム数
        self.best_win_ratio = 0.0  # 最善モデルの勝率
        
        # Loss記録用
        self.loss_history = []  # Loss値の履歴
        self.entropy_history = []  # エントロピーの履歴
        
        # 並列処理用
        self.num_cpu_workers = multiprocessing.cpu_count() - 1  # CPU並列数
        self.data_augmentation_executor = ThreadPoolExecutor(max_workers=self.num_cpu_workers)
        
        # モデルの初期化
        # policy_value_netを初期化 (PolicyValueNetはポリシーとバリューネットワークを統合したクラスと仮定)
        self.policy_value_net = PolicyValueNet(self.board_size,env=self.env)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                       c_puct=self.c_puct, n_playout=self.n_playout,
                                       is_selfplay=True)
        
        # GPU使用率を制限するための設定
        if torch.cuda.is_available():
            # GPUメモリの使用量を制限
            torch.cuda.set_per_process_memory_fraction(0.8)  # 80%に制限
            torch.cuda.empty_cache()
            
            # GPU計算のバッチサイズを動的に調整
            self.gpu_batch_size = 32  # GPUでの推論バッチサイズを小さくする
            print(f"GPU使用率を80%に制限しました")
        else:
            self.gpu_batch_size = 64
    
    def __del__(self):
        """デストラクタでスレッドプールを終了"""
        if hasattr(self, 'data_augmentation_executor'):
            self.data_augmentation_executor.shutdown(wait=True)
    def get_equi_data(self, play_data):
        """
        収集したセルフプレイデータを回転や反転によって拡張します。
        これにより、モデルの汎用性を向上させます。
        CPUで並列実行します。
        """
        # 並列処理のためのタスクを準備
        tasks = []
        for state, mcts_prob, winner in play_data:
            tasks.append((state, mcts_prob, winner, self.board_size))
        
        # CPUで並列実行
        extend_data = []
        with ThreadPoolExecutor(max_workers=self.num_cpu_workers) as executor:
            futures = [executor.submit(parallel_data_augmentation, task) for task in tasks]
            
            for future in as_completed(futures):
                extend_data.extend(future.result())
        
        return extend_data
    # セルフプレイデータを収集するメソッド
    def collect_selfplay_data(self, num_games):
        """
        MCTSプレイヤーを用いた自己対戦をシミュレートし、学習データを収集します。
        収集したデータは、回転や反転によって拡張（オーグメンテーション）されます。
        """
        for i in range(num_games):
                winner, play_data = self.game.start_self_play(self.mcts_player,
                                                            temp=self.temp)
                play_data = list(play_data)[:]
                print("play_data winner values:", [d[2] for d in play_data])

                self.episode_len = len(play_data)
                # データの拡張
                play_data = self.get_equi_data(play_data)
                
                self.date_buffer.extend(play_data)
    # ポリシーを更新するメソッド
    def policy_update(self):
        """
        収集したデータからミニバッチを作成し、ポリシーとバリューネットワークを更新します。
        """
        # 経験再生バッファからミニバッチをサンプリング（CPUで実行）
        mini_batch = random.sample(self.date_buffer, self.batch_size)
        
        # CPUで並列処理でデータを準備
        with ThreadPoolExecutor(max_workers=self.num_cpu_workers) as executor:
            # バッチデータを並列で処理
            futures = []
            batch_size_per_thread = len(mini_batch) // self.num_cpu_workers
            
            for i in range(self.num_cpu_workers):
                start_idx = i * batch_size_per_thread
                if i == self.num_cpu_workers - 1:
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
        
        # エポック数だけトレーニングを繰り返す
        for i in range(self.epochs):
            # ネットワークを1ステップ学習させる
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            
            # 更新後のポリシーとバリューを取得
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            
            # 更新前後のポリシーのKLダイバージェンスを計算（CPUで実行）
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            # KLが目標値の4倍を超えたら、学習を早期終了
            if kl > self.kl_targ * 4:  
                break
            
            # KLダイバージェンスに基づいて学習率を調整
            if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
            elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
            
            # 学習の進捗を評価するための指標を計算（CPUで実行）
            # 0による除算を防ぐために分母をチェック
            winner_var = np.var(np.array(winner_batch))
            if winner_var > 1e-8:  # 分母が0に近くないことを確認
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
            print(("kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.3f},"
                "explained_var_new:{:.3f}"
                ).format(kl,
                            self.lr_multiplier,
                            loss,
                            entropy,
                            explained_var_old,
                            explained_var_new))
        # 損失とエントロピーを返す
        # Loss値を記録
        self.loss_history.append(loss)
        self.entropy_history.append(entropy)
        
        return loss, entropy
    
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
        AlphaZeroのトレーニングパイプラインを実行します。
        自己対戦データの収集、ポリシーの更新、モデルの評価を繰り返します。
        """
        try:
            import time
            # トレーニング開始時に初期の最善モデルを保存（存在しない場合）
            import os
            if not os.path.exists('./best_policy.model'):
                print("初期の最善ポリシーモデルを保存しています...")
                self.policy_value_net.save_model('./best_policy.model')
                print("初期の最善ポリシーモデルを保存しました。")
            
            # 指定されたゲームバッチ数だけトレーニングサイクルを繰り返す
            for i in range(self.game_batch_num):
                cycle_start_time = time.time()
                
                # 自己対戦データを収集する
                selfplay_start = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                selfplay_time = time.time() - selfplay_start
                
                print(f"ゲーム {i+1}/{self.game_batch_num} 完了。自己対戦時間: {selfplay_time:.2f}秒")
                
                # バッファに十分なデータが溜まったらポリシーを更新する
                if len(self.date_buffer) >= self.batch_size:
                    update_start = time.time()
                    loss, entropy = self.policy_update()
                    update_time = time.time() - update_start
                    print(f"ポリシー更新時間: {update_time:.2f}秒")
                    self.save_loss_graph()
                
                cycle_time = time.time() - cycle_start_time
                print(f"サイクル {i+1} 合計時間: {cycle_time:.2f}秒")
                print("-" * 50)
                
                # 一定の頻度で現在のモデルを評価する
                if (i+1) % self.check_freq == 0:
                    print(f"現在の自己対戦バッチ: {i+1}")
                    eval_start = time.time()
                    win_ratio = self.policy_evaluate()
                    eval_time = time.time() - eval_start
                    print(f"モデル評価時間: {eval_time:.2f}秒")
                    # 現在のポリシーを保存
                    self.policy_value_net.save_model('./current_policy.model')
                    
                    # Lossグラフを保存
                    self.save_loss_graph()
                    
                    # 新しいモデルが最善モデルを上回った場合（勝率55%以上）
                    if win_ratio > max(self.best_win_ratio, 0.51): 
                        print(f"新しい最善ポリシーが見つかりました！（勝率: {win_ratio:.3f}）")
                        self.best_win_ratio = win_ratio
                        # 最善ポリシーを更新して保存
                        self.policy_value_net.save_model('./best_policy.model')
        except KeyboardInterrupt:
            print('\n\rトレーニングを中断しました。')
            # 最終的なグラフを保存
            self.save_loss_graph()
            
    def policy_evaluate(self, n_games=10):
        """
        現在のポリシーと最善のポリシーを比較評価します。
        n_games回対戦し、現在のポリシーの勝率を計算します。
        """
        # 現在のポリシーを持つMCTSプレイヤーを作成（わずかに探索を増やす）
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout + 20,  # わずかに多く探索
                                         is_selfplay=False)
        # 最善のポリシーをロードしてMCTSプレイヤーを作成
        best_policy = PolicyValueNet(self.board_size, env=self.env)
        try:
            # 最善のモデルをロード
            import os
            if os.path.exists('./best_policy.model'):
                best_policy.load_model('./best_policy.model')
                print("最善のポリシーモデルをロードしました。")
            else:
                # モデルが存在しない場合は、現在のモデルを最善モデルとして保存
                print("最善のポリシーモデルが見つかりません。現在のモデルを最善として保存します。")
                self.policy_value_net.save_model('./best_policy.model')
                # 初回の場合は勝率を0.55として返し、現在のモデルが更新されるようにする
                return 0.55
        except Exception as e:
            # モデルのロードに失敗した場合
            print(f"最善のポリシーモデルのロードに失敗しました: {e}")
            # 現在のモデルを最善モデルとして保存
            self.policy_value_net.save_model('./best_policy.model')
            return 0.55

        best_mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,  # 標準の探索回数
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
            if winner == 1:  # 現在のプレイヤー(player1)が勝利
                win_cnt += 1
                current_wins_as_first += 1
            elif winner == 0:  # 引き分け
                draw_cnt += 1
            print(f"先手戦 {i+1}/{n_games//2}: 勝者 = {winner}")
        
        # n_games/2 回、現在のプレイヤーが後手で対戦
        for i in range(n_games // 2):
            winner = self.game.start_play(best_mcts_player,
                                          current_mcts_player,
                                          start_player=0,
                                          is_shown=0)
            if winner == -1:  # 現在のプレイヤー(player2)が勝利
                win_cnt += 1
                current_wins_as_second += 1
            elif winner == 0:  # 引き分け
                draw_cnt += 1
            print(f"後手戦 {i+1}/{n_games//2}: 勝者 = {winner}")
        
        win_ratio = win_cnt / n_games
        print(f"詳細結果:")
        print(f"  先手での勝利: {current_wins_as_first}/{n_games//2}")
        print(f"  後手での勝利: {current_wins_as_second}/{n_games//2}")
        print(f"  引き分け: {draw_cnt}/{n_games}")
        print(f"  総合結果: {win_cnt}勝 / {n_games}戦, 勝率: {win_ratio:.3f}")
        return win_ratio
    
    def _simulate(self, data):
        # Simulate the game using the model and update the policy and value networks
        pass

    def play(self, state):
        # Use the model to select the best move based on the current state
        pass