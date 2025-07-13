import numpy as np
from GomokuEnv import GomokuEnv

class Game(object):
    def __init__(self, env:GomokuEnv, board_size=19):
        self.env = env
        self.board_size = env.board_size
        self.board = env.board
        self.current_player = env.current_player
        self.lastmove = env.lastmove
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
            #TODO:ボードがあっているか確認
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
            