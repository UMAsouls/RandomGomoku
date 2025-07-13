import numpy as np
from GomokuEnv import GomokuEnv

class Game(object):
    def __init__(self, env:GomokuEnv, board_size=19):
        self.env = env
        self.board_size = env.board_size
        self.board = env.board
        self.current_player = env.current_player
        self.lastmove = env.lastmove

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        2人のプレイヤー間でゲームを開始します。
        player1, player2: プレイヤーオブジェクト
        start_player: 先手プレイヤー（0=player1, 1=player2）
        is_shown: ゲームの進行状況を表示するかどうか
        """
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                           'or 1 (player2 first)')
        
        # 新しいゲーム環境を初期化
        self.env = GomokuEnv(board_size=self.board_size)
        
        # start_playerに基づいてプレイヤーの割り当てを決定
        if start_player == 0:
            # player1が先手（プレイヤー1）
            p1, p2 = 1, 2
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)
            players = {p1: player1, p2: player2}
        else:
            # player2が先手（プレイヤー1）
            p1, p2 = 1, 2
            player1.set_player_ind(p2)
            player2.set_player_ind(p1)
            players = {p1: player2, p2: player1}
        
        if is_shown:
            print(f"Game start: Player {p1} vs Player {p2}")
            print(f"先手: Player {p1 if start_player == 0 else p2}")
        

        
        # ゲームループ
        move_count = 0
        max_moves = self.board_size * self.board_size  # 最大手数制限
        done = False
        while not done and move_count < max_moves:
            # 現在のプレイヤーを決定
            current_player = self.env.current_player
            
            if is_shown:
                print(f"\n--- 手番 {move_count + 1}: Player {current_player} ---")
            
            # プレイヤーから行動を取得
            if current_player in players:
                try:
                    move = players[current_player].get_action(self.env)
                    if is_shown:
                        print(f"Player {current_player} の手: {move}")
                except Exception as e:
                    if is_shown:
                        print(f"Player {current_player} でエラーが発生: {e}")
                    # エラーが発生した場合、相手の勝利とする
                    if current_player == 1:
                        return -1  # Player 2の勝利
                    else:
                        return 1   # Player 1の勝利
            else:
                if is_shown:
                    print(f"不正なプレイヤー: {current_player}")
                break
            
            # 行動を実行
            try:
                _, reward, done, _ = self.env.step(move)
                move_count += 1
                
                if is_shown and hasattr(self.env.board, 'PrintBoard'):
                    self.env.board.PrintBoard()
                
            except Exception as e:
                if is_shown:
                    print(f"無効な手: {move}, エラー: {e}")
                # 無効な手を打った場合、相手の勝利とする
                if current_player == 1:
                    return -1  # Player 2の勝利
                else:
                    return 1   # Player 1の勝利
        
        # ゲーム終了処理
        if done:
            self.env.board.PrintBoard()
            if is_shown:
                if reward == 1:
                    print(f"\nゲーム終了: 先手（Player 1）の勝利!")
                elif reward == -1:
                    print(f"\nゲーム終了: 後手（Player 2）の勝利!")
                else:
                    print("\nゲーム終了: 引き分け!")
            
            # AlphaZeroの評価に合わせた勝敗判定を返す
            if start_player == 0:
                # player1が先手の場合
                if reward == 1:
                    return 1  # player1（現在のモデル）の勝利
                elif reward == -1:
                    return -1  # player2（最善のモデル）の勝利
                else:
                    return 0  # 引き分け
            else:
                # player2が先手の場合
                if reward == 1:
                    return -1  # player2（最善のモデル）の勝利
                elif reward == -1:
                    return 1  # player1（現在のモデル）の勝利
                else:
                    return 0  # 引き分け
        else:
            # 最大手数に達した場合は引き分け
            if is_shown:
                print(f"\nゲーム終了: 最大手数 {max_moves} に達しました。引き分け!")
            return 0
        
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """
        セルフプレイを開始します。
        player: MCTSプレイヤー
        is_shown: ゲームの進行状況を表示するかどうか
        temp: 温度パラメータ（探索のランダム性を制御）
        """
        self.env = GomokuEnv(board_size=self.board_size)
        states, mcts_probs, current_players = [], [], []
        while True:
            move, mcts_prob = player.get_action(self.env, temp=temp, return_prob=True)
            
            # ボードが満杯で有効な手がない場合の処理
            if move is None:
                # 引き分け処理
                winners_z = np.zeros(len(current_players))
                player.reset_player()
                return winners_z, zip(states, mcts_probs, winners_z)
            
            #TODO:ボードがあっているか確認
            # print(self.env.board.GetBoardInt())
            states.append(self.env.board.GetBoardInt())
            mcts_probs.append(mcts_prob)
            current_players.append(self.env.current_player)
            
            _,reward,done,_= self.env.step(move)
            if done:
                winners_z = np.zeros(len(current_players))
                if reward != -1:
                    winners_z[np.array(current_players) == self.env.train_player] = 1
                    winners_z[np.array(current_players) != self.env.train_player] = -1
                player.reset_player()
                return winners_z, zip(states, mcts_probs, winners_z)
            