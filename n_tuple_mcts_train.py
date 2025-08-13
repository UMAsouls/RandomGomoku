
from N_Tuple import NTupleMCTSAgent, MCTSReplayBuffer

from NTupleGomokuEnv import NTupleGomokuEnv

BOARD_SIZE = 9  # ボードのサイズ

MODEL_DIR = "NTupleMCTSModel"
MODEL_NAME = "S9Model8_13_3"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

NETS = [5]

REP_BUFFER_SIZE = 100000
BATCH_SIZE = 100

EPOCHS = 10
EPISODES = 100
MAX_EPISODES = 1000000

CPUCT = 5.0
SEARCH_TIME = 0.04

class NTupleMCTSTrainer:
    def __init__(self):
        self.env = NTupleGomokuEnv(BOARD_SIZE, "both")
        self.agent = NTupleMCTSAgent(self.env, BOARD_SIZE, MODEL_PATH, NETS, CPUCT, SEARCH_TIME)
        self.replay_buffer = MCTSReplayBuffer(REP_BUFFER_SIZE, BATCH_SIZE, BOARD_SIZE)
        
        self.epochs = EPOCHS
        self.board_size = BOARD_SIZE
        self.batch_size = BATCH_SIZE
        self.episodes = EPISODES
        
        self.max_episodes = MAX_EPISODES
        
    def run(self):
        epi = 0
        while epi < self.max_episodes:
            for i in range(self.episodes):
                print(f"--- Self-Play Episode {epi+1}/{self.max_episodes} ---")
            
                # 1. 自己対戦を行い、結果をReplayBufferに追加する
                self.run_episode()
                
                epi += 1
        
                
            for i in range(self.epochs):
                if len(self.replay_buffer) > self.batch_size:
                    print(f"--- Training Step {i+1}/{self.epochs} ---")
                    self.train_step()
                
            # 3. 定期的にモデルを保存する
            print("Saving models...")
            self.agent.save()
                
                
        return self.episodes
        
        
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
            self.agent.policy_train(board_state, target_policy)
            self.agent.value_train(board_state, action, target_value) 
            
            
if __name__ == "__main__":
    trainer = NTupleMCTSTrainer()
    trainer.run()