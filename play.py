from dqn import DQNAgent
import GomokuEnv
from  agent import RandomAgent
from GomokuEnv import Stone

random_agent = RandomAgent()
dqn_agent = DQNAgent()
env = GomokuEnv.GomokuEnv(train_target="first")


save_path = './dqn_model'
print('Model Path:', save_path)
# === 学習済みモデルでGomokuをプレイ ===
dqn_agent.load(save_path)  # モデルをロード
dqn_agent.epsilon = 0  # greedy policy
state = env.board.copy()
done = False
total_reward = 0
print('Initial State:')
#ai vs human
# while not done:
#     if env.current_player == env.train_player:
#         action = dqn_agent.get_action(state)  # DQNエージェントのターン
#     else:
#         action = env.get_human_action()
#     next_state, reward, done, info = env.step(action)
#     env.current_player = 3 - env.current_player
#     state = next_state
#     total_reward += reward
#     print('Total Reward:', total_reward)
#     print('Current State:')
#     env.render()
#     if done:
#         break

#ai vs random

while not done:
    if env.current_player == env.train_player:
        action = dqn_agent.get_action(state)  # DQNエージェントのターン
    else:
        action = random_agent.get_action(state)
        
    
    next_state, reward, done, info = env.step(action)
    
    env.current_player = 3 - env.current_player
    
    
    state = next_state
    total_reward += reward
    print('Total Reward:', total_reward)
    print('Current State:')
    env.render()
   
    
    
    
print('Total Reward:', total_reward)