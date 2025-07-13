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
        if start_player not in (0, 1):
         raise Exception('start_player should be either 0 (player1 first) '
                        'or 1 (player2 first)')
        self.env = GomokuEnv(board_size=self.board_size)
        p1, p2 = 1,2
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            print(f"Game start: Player {p1} vs Player {p2}")
            # self.env.board.PrintBoard()
        _, reward, done, _ = self.env.step(None)  # 初期化
        if done:
            if is_shown:
                if reward == 1:
                    print(f"Player {p1} wins!")
                elif reward == -1:
                    print(f"Player {p2} wins!")
                else:
                    print("It's a draw!")
            return reward
            
        
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
            