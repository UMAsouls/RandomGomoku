"""
逐次処理版AlphaZero（比較用）
"""
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

BOARD_SIZE = 8
N_IN_ROW = 5

class AlphaZeroSequential:
    """
    逐次処理版AlphaZero（並列処理版との比較用）
    """
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.game = Game(self.env, board_size=BOARD_SIZE)
        self.n_in_row = N_IN_ROW
        
        # トレーニング用パラメータ
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 20
        self.kl_targ = 0.02
        self.check_freq = 100
        self.game_batch_num = 5000
        self.best_win_ratio = 0.0
        
        # モデルの初期化
        self.policy_value_net = PolicyValueNet(self.board_size, env=self.env)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                     c_puct=self.c_puct, n_playout=self.n_playout,
                                     is_selfplay=True)
        
        # GPU使用率を制限
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.cuda.empty_cache()
    
    def get_equi_data(self, play_data):
        """
        データ拡張を逐次で実行
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            state = np.array(state)
            mcts_prob = np.array(mcts_prob)
            
            # stateが1次元の場合、4次元形状に変換
            if len(state.shape) == 1:
                state = state.reshape(4, self.board_size, self.board_size)
            elif len(state.shape) == 2:
                temp_state = np.zeros((4, self.board_size, self.board_size))
                temp_state[0] = state
                state = temp_state
            
            # mcts_probを2次元に変換
            mcts_prob_2d = mcts_prob.reshape(self.board_size, self.board_size)
            
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
    
    def collect_selfplay_data(self, num_games):
        """
        逐次で自己対戦データを収集
        """
        all_play_data = []
        
        # 逐次実行
        for i in range(num_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)
            # データ拡張
            extended_data = self.get_equi_data(play_data)
            all_play_data.extend(extended_data)
        
        return all_play_data
