
from N_Tuple import NTupleMCTSAgent, MCTSReplayBuffer

from NTupleGomokuEnv import NTupleGomokuEnv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

BOARD_SIZE = 9  # ボードのサイズ

MODEL_DIR = "NTupleMCTSModel"
MODEL_NAME = "S9Model8_18_1"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME
GRAPH_PATH = "pv_loss.png"

NETS = [5]

REP_BUFFER_SIZE = 10000
BATCH_SIZE = 100

EPOCHS = 5
EPISODES = 100
MAX_EPISODES = 100000

PV_SAVERATE = 10

CPUCT = 1.0
#SEARCH_TIME = 0.1

SIMULATION_TIME = 200

LEARNING_RATE = 0.01

class NTupleMCTSTrainer:
    def __init__(self):
        self.env = NTupleGomokuEnv(BOARD_SIZE, "both")
        self.agent = NTupleMCTSAgent(self.env, BOARD_SIZE, MODEL_PATH, NETS, CPUCT, SIMULATION_TIME, LEARNING_RATE)
        self.replay_buffer = MCTSReplayBuffer(REP_BUFFER_SIZE, BATCH_SIZE, BOARD_SIZE)
        
        self.epochs = EPOCHS
        self.board_size = BOARD_SIZE
        self.batch_size = BATCH_SIZE
        self.episodes = EPISODES
        
        self.max_episodes = MAX_EPISODES
        
        self.p_errors = [0]
        self.v_errors = [0]
        
        self.e_idx = 0
        
        self.pv_saverate = PV_SAVERATE
        
        plt.rcParams["font.size"] = 12
        
        
                
    def run(self):
        epi = 0
        while epi < self.max_episodes:
            print(f"--- Self-Play Episode {epi+1}/{self.max_episodes} ---")
            # 1. 自己対戦を行い、結果をReplayBufferに追加する
            self.run_episode()
            epi += 1 
            
            for i in range(self.epochs):
                if len(self.replay_buffer) > self.batch_size:
                    print(f"--- Training Step {i+1}/{self.epochs} ---")
                    self.train_step()
            
            if(epi % self.episodes == 0):  
                # 定期的にモデルを保存する
                print("Saving models...")
                self.agent.save()
                
            if(epi % self.pv_saverate == 0): 
                self.save_loss()
                
                
                
        return self.episodes
    
    def save_loss(self):
        self.p_errors[self.e_idx] /=  self.epochs*self.batch_size*self.pv_saverate
        self.v_errors[self.e_idx] /=  self.epochs*self.batch_size*self.pv_saverate
        
        fig= plt.figure(figsize=(15, 5))
        fig.suptitle(f'PV_loss')
        
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        # Policy Loss
        ax1.plot(self.p_errors, color="blue", label="Policy Loss")
        ax1.set_xlabel("Training Epochs")
        ax1.set_ylabel("Average Loss")
        ax1.set_title("Policy Network Loss")
        ax1.legend()
        ax1.grid(True)

        # Value Loss
        ax2.plot(self.v_errors, color="red", label="Value Loss")
        ax2.set_xlabel("Training Epochs")
        ax2.set_ylabel("Average Loss")
        ax2.set_title("Value Network Loss")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_dir = os.path.dirname(f"{MODEL_PATH}/{GRAPH_PATH}")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(f"{MODEL_PATH}/{GRAPH_PATH}")
        
        plt.close(fig) # メモリ解放のために図を閉じる
        print(f"Loss plot saved to {MODEL_PATH}/{GRAPH_PATH}")
        
        self.e_idx += 1
        self.p_errors.append(0)
        self.v_errors.append(0)
        
        
    def run_episode(self):
        self.env.reset()
        game_history = []
        
        done = False
        reward = 0
        
        current_player: int
        
        self.agent.ResetMemo()
        while not done:
            # 現在の盤面、プレイヤー情報を取得
            board_state = self.env.GetBoard_CurrentPlayer()
            current_player = self.env.current_player
            
            # MCTSで手と「思考のログ（訪問回数の分布）」を取得
            action, move_probs = self.agent.Search()
            
            # あとで学習に使うために、履歴を保存
            game_history.append([board_state, action, current_player, move_probs])
            
            # 実際に着手
            (actx,acty) = (action%self.board_size, action//self.board_size)
            _, reward, done, _ = self.env.step((actx,acty))
            #self.env.Animation()

        #self.env.AnimationEnd()
        winner = current_player
        
        print("episode end")
        self.env.PrintBoard()
        self.agent.PrintMemo()
        
        for board_state, action, player, move_probs in game_history:
            game_value = 1 if player == winner else -1
            # バッファに追加
            self.replay_buffer.add(board_state, action, move_probs, game_value)
            
    def train_step(self):
        """ReplayBufferからバッチを取得してネットワークを学習"""
        # バッファからランダムにミニバッチを取得
        mini_batch = self.replay_buffer.get_batch()
        
        idx = 0
        for board_state, action, target_policy, target_value in mini_batch:
            p_error = self.agent.policy_train(board_state, target_policy)
            v_error = self.agent.value_train(board_state, action, target_value)
            
            self.p_errors[self.e_idx] += p_error
            self.v_errors[self.e_idx] += v_error
         
            
            
if __name__ == "__main__":
    trainer = NTupleMCTSTrainer()
    trainer.run()