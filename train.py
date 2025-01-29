import matplotlib.pyplot as plt
import numpy as np
import GomokuEnv
import RandomGomoku
from dqn import DQNAgent
from dqn import RandomAgent


# === Main ===
episodes = 300
sync_interval = 20
## TODO: RandomGomoku.makeEnv() is not defined
env = GomokuEnv.GomokuEnv()
dqn_agent = DQNAgent()
random_agent = RandomAgent()
reward_history = []
first_win = 0
second_win = 0

# 学習の実行
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:

        if env.current_player == 1:
            action = dqn_agent.get_action(state)  # DQNエージェントのターン
        else:
            action = random_agent.get_action(state)  # ランダムエージェントのターン
        
        
        next_state, reward, done, info = env.step(action)
        
        
        if(done):
            if(env.current_player !=1):
                first_win += 1
            else:
                second_win += 1

        dqn_agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        dqn_agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 1 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))
        print("first win : {}, second win : {},percentage : {}".format(first_win, second_win, first_win/(first_win+second_win)))

# モデルの保存
save_path = './dqn_model'
dqn_agent.save(save_path)

# === Plot ===
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()