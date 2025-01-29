import matplotlib.pyplot as plt
import numpy as np
import GomokuEnv
import RandomGomoku
from dqn import DQNAgent
from dqn import RandomAgent
import GomokuEnv
from GomokuEnv import Stone

save_path = './dqn_model'


# === Main ===
episodes = 50000
sync_interval = 20
env = None


#初期エージェントの設定
train_agent = DQNAgent()
opponent_agent = RandomAgent()

reward_history = []
persentage_history = []
train_target_win = 0

opponent_win = 0

# 学習の実行
for episode in range(episodes):
    
    if episode % 500 ==0 and episode != 0:
        train_agent.load(save_path)
        opponent_agent = DQNAgent()
        opponent_agent.load(save_path)
    
    # 先手後手を交互に設定
    if episode % 2 == 0:
        env = GomokuEnv.GomokuEnv(stone=Stone.BLACK)  # 黒（先行）
    else:
        env = GomokuEnv.GomokuEnv(stone=Stone.WHITE)  # 白（後行）
    
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if env.current_player == env.train_player:
            action = train_agent.get_action(state)  # DQNエージェントのターン
        else:
            action = opponent_agent.get_action(state)  # ランダムエージェントのターン
        
        next_state, reward, done, info = env.step(action)
        
        if done:
            if env.current_player != env.train_player:
                train_target_win += 1
            else:
                
                opponent_win += 1

        train_agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        train_agent.sync_qnet()

    reward_history.append(total_reward)
    persentage_history.append(train_target_win/(train_target_win+opponent_win))
    if episode % 1 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))
        print("train_target : {}, opponent win : {}, percentage : {}".format(train_target_win, opponent_win, train_target_win/(train_target_win+opponent_win)))

# モデルの保存
train_agent.save(save_path)

# === Plot ===
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()

plt.xlabel('Episode')
plt.ylabel('Winning Percentage')
plt.plot(range(len(persentage_history)), persentage_history)
plt.show()
