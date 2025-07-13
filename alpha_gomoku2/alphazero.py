import numpy as np
from GomokuEnv import GomokuEnv
from game import Game
import torch
from collections import deque
import random
import torch.nn.functional as F
from network import PolicyValueNet
from mcts import MCTSPlayer
import random
BOARD_SIZE = 8  # ボードサイズ
N_IN_ROW = 5 # 勝利条件（連続する石の数）

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
        self.n_playout = 400  # 各着手ごとのプレイアウト回数
        self.c_puct = 5  # UCBスコアの探索項の係数
        self.buffer_size = 10000  # 経験再生バッファのサイズ
        self.batch_size = 512  # トレーニング時のバッチサイズ
        self.date_buffer = deque(maxlen=self.buffer_size)  # 経験再生バッファ
        self.play_batch_size = 1  # 自己対戦の並列実行数
        self.epochs = 20  # 各更新ステップでのエポック数
        self.kl_targ = 0.02 # KLダイバージェンスの目標値
        self.check_freq = 50  # モデル評価の頻度
        self.game_batch_num = 1500  # 1回のトレーニングサイクルでプレイするゲーム数
        self.best_win_ratio = 0.0  # 最善モデルの勝率
        
        # モデルの初期化
        # policy_value_netを初期化 (PolicyValueNetはポリシーとバリューネットワークを統合したクラスと仮定)
        self.policy_value_net = PolicyValueNet(self.board_size,env=self.env)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                       c_puct=self.c_puct, n_playout=self.n_playout,
                                       is_selfplay=True)
    def get_equi_data(self, play_data):
        """
        収集したセルフプレイデータを回転や反転によって拡張します。
        これにより、モデルの汎用性を向上させます。
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_size)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
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
                self.episode_len = len(play_data)
                # データの拡張
                play_data = self.get_equi_data(play_data)
                self.date_buffer.extend(play_data)
    # ポリシーを更新するメソッド
    def policy_update(self):
        """
        収集したデータからミニバッチを作成し、ポリシーとバリューネットワークを更新します。
        ミニバッチ：経験再生バッファからランダムに抽出されたデータの一部。
                    一度に全てのデータを使うのではなく、バッチ単位で学習することで、
                    計算効率を高め、学習を安定させます。
        KLダイバージェンス：2つの確率分布の差異を測る指標。
                         ここでは、更新前後のポリシーの出力（着手確率）の差を測ります。
                         この値が大きくなりすぎないように学習率を動的に調整し、
                         ポリシーが急激に変化して学習が不安定になるのを防ぎます。
        """
        # 経験再生バッファからミニバッチをサンプリング
        mini_batch = random.sample(self.date_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
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
            
            # 更新前後のポリシーのKLダイバージェンスを計算
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
            
            # 学習の進捗を評価するための指標を計算
            explained_var_old = (1 -
                         np.var(np.array(winner_batch) - old_v.flatten()) /
                         np.var(np.array(winner_batch)))
            explained_var_new = (1 -
                                np.var(np.array(winner_batch) - new_v.flatten()) /
                                np.var(np.array(winner_batch)))
            
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
        return loss, entropy
    
    def train(self):
        """
        AlphaZeroのトレーニングパイプラインを実行します。
        自己対戦データの収集、ポリシーの更新、モデルの評価を繰り返します。
        """
        try:
            # 指定されたゲームバッチ数だけトレーニングサイクルを繰り返す
            for i in range(self.game_batch_num):
                # 自己対戦データを収集する
                self.collect_selfplay_data(self.play_batch_size)
                print(f"ゲーム {i+1}/{self.game_batch_num} 完了。")
                
                # バッファに十分なデータが溜まったらポリシーを更新する
                if len(self.date_buffer) >= self.batch_size:
                    loss, entropy = self.policy_update()
                
                # 一定の頻度で現在のモデルを評価する
                if (i+1) % self.check_freq == 0:
                    print(f"現在の自己対戦バッチ: {i+1}")
                    win_ratio = self.policy_evaluate()
                    # 現在のポリシーを保存
                    self.policy_value_net.save_model('./current_policy.model')
                    
                    # 新しいモデルが最善モデルを上回った場合
                    if win_ratio > self.best_win_ratio:
                        print("新しい最善ポリシーが見つかりました！")
                        self.best_win_ratio = win_ratio
                        # 最善ポリシーを更新して保存
                        self.policy_value_net.save_model('./best_policy.model')
        except KeyboardInterrupt:
            print('\n\rトレーニングを中断しました。')
            
    def policy_evaluate(self, n_games=10):
        """
        現在のポリシーと最善のポリシーを比較評価します。
        n_games回対戦し、現在のポリシーの勝率を計算します。
        """
        # 現在のポリシーを持つMCTSプレイヤーを作成
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout,
                                         is_selfplay=False)
        # 最善のポリシーをロードしてMCTSプレイヤーを作成
        best_policy = PolicyValueNet(self.board_size, env=self.env)
        try:
            # 最善のモデルをロード
            import os
            if os.path.exists('./best_policy.model'):
                best_policy.load_model('./best_policy.model')
            else:
                # モデルが存在しない場合は、勝率を0として扱い、現在のモデルが最善となるようにする
                print("最善のポリシーモデルが見つかりません。現在のモデルを最善として保存します。")
                return 0.0
        except Exception as e:
            # モデルのロードに失敗した場合
            print(f"最善のポリシーモデルのロードに失敗しました: {e}")
            return 0.0

        best_mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=False)

        win_cnt = 0
        # n_games/2 回、現在のプレイヤーが先手で対戦
        for i in range(n_games // 2):
            winner = self.game.start_play(current_mcts_player,
                                          best_mcts_player,
                                          start_player=0,
                                          is_shown=0)
            if winner == 1: # 現在のプレイヤー(player1)が勝利
                win_cnt += 1
        
        # n_games/2 回、現在のプレイヤーが後手で対戦
        for i in range(n_games // 2):
            winner = self.game.start_play(best_mcts_player,
                                          current_mcts_player,
                                          start_player=0,
                                          is_shown=0)
            if winner == -1: # 現在のプレイヤー(player2)が勝利
                win_cnt += 1
        
        win_ratio = win_cnt / n_games
        print(f"対戦結果: {win_cnt}勝 / {n_games}戦, 勝率: {win_ratio}")
        return win_ratio
    
    def _simulate(self, data):
        # Simulate the game using the model and update the policy and value networks
        pass

    def play(self, state):
        # Use the model to select the best move based on the current state
        pass