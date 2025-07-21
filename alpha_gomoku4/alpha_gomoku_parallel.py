import concurrent.futures
from alphazero import AlphaZero

class ParallelAlphaZero(AlphaZero):
    def collect_selfplay_data_parallel(self, num_games):
        """
        複数試合を並列で自己対戦し、データを収集して拡張する。
        各試合ごとに独立したGomokuEnv/Gameを使う。
        """
        from GomokuEnv import GomokuEnv
        from game import Game
        from mcts import MCTSPlayer
        play_data_list = []

        def run_self_play():
            # 各スレッドで独立した環境・ゲーム・MCTSPlayerを生成
            env = GomokuEnv(board_size=self.board_size)
            game = Game(env, board_size=self.board_size)
            mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                     c_puct=self.c_puct, n_playout=self.n_playout,
                                     is_selfplay=True)
            winner, play_data = game.start_self_play(mcts_player, temp=self.temp)
            return winner, play_data

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.play_batch_size) as executor:
            futures = [executor.submit(run_self_play) for _ in range(num_games)]
            for future in concurrent.futures.as_completed(futures):
                winner, play_data = future.result()
                play_data = list(play_data)[:]
                play_data_list.extend(play_data)
        # データ拡張
        extended_data = self.get_equi_data(play_data_list)
        self.date_buffer.extend(extended_data)

    def train_parallel(self):
        """
        並列で複数試合を行い、まとめて学習するトレーニングパイプライン。
        """
        import time
        import os
        if not os.path.exists('./best_policy.model'):
            print("初期の最善ポリシーモデルを保存しています...")
            self.policy_value_net.save_model('./best_policy.model')
            print("初期の最善ポリシーモデルを保存しました。")

        for i in range(self.game_batch_num):
            cycle_start_time = time.time()
            # 並列で自己対戦データ収集
            selfplay_start = time.time()
            self.collect_selfplay_data_parallel(self.play_batch_size)
            selfplay_time = time.time() - selfplay_start
            print(f"ゲーム {i+1}/{self.game_batch_num} 完了。自己対戦(並列)時間: {selfplay_time:.2f}秒")

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
                self.policy_value_net.save_model('./current_policy.model')
                self.save_loss_graph()
                if win_ratio > max(self.best_win_ratio, 0.51):
                    print(f"新しい最善ポリシーが見つかりました！（勝率: {win_ratio:.3f}）")
                    self.best_win_ratio = win_ratio
                    self.policy_value_net.save_model('./best_policy.model')

if __name__ == "__main__":
    # 並列AlphaZeroのインスタンス化
    agent = ParallelAlphaZero()
    # 並列トレーニング開始
    agent.train_parallel()