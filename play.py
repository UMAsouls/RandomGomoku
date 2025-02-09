from dqn import DQNAgent
import GomokuEnv
from  agent import RandomAgent
from GomokuEnv import Stone
from agent import RuleBasedAgent
from agent import MinimaxAgent

random_agent = RandomAgent()
dqn_agent = DQNAgent()
rule_based_agent = RuleBasedAgent()
minimax_agent = MinimaxAgent()
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

#ai vs random
# env.render()

# while not done:
#     if env.current_player == env.train_player:
#         action = dqn_agent.get_action(state)  # DQNエージェントのターン
#     else:
#         action = random_agent.get_action(state)
        
    
#     next_state, reward, done, info = env.step(action)
    
#     env.current_player = 3 - env.current_player
    
    
#     state = next_state
#     total_reward += reward
#     print('Total Reward:', total_reward)
#     print('Current State:')
#     env.render()
   



#ai vs rule_based
# env.render()
# while not done:
#     if env.current_player == 1:
#         print('Rule Based Agent')
        
#         action = rule_based_agent.get_action(state,env.current_player)
#     else:
#         print('minimax Agent')
#         action = minimax_agent.get_action(state,env.current_player)  # DQNエージェントのターン
#     print("state",state)
#     next_state, reward, done, info = env.step(action)
#     env.current_player = 3 - env.current_player
#     state = next_state
#     total_reward += reward
#     print('Total Reward:', total_reward)
#     print('Current State:')
#     env.render()
#     if done:
#         print('Game Over')
#         if env.current_player == 2:
#             print('first Win')
#         else:
#             print('second Win')
#         break
# minimax vs rule_based
first_win = 0
second_win = 0
episodes = 50

for episode in range(episodes):
    
    env = GomokuEnv.GomokuEnv(train_target="first" if episode % 2 == 0 else "second")

    state = env.board.copy()
    done = False
    env.render()
    while not done:
        if env.current_player == env.train_player:
            print('minimax Agent')
            action = minimax_agent.get_action(state,env.current_player)
        else:
            print('rule_based Agent')
            action = rule_based_agent.get_action(state,env.current_player)  # DQNエージェントのターン
        next_state, reward, done, info = env.step(action)
        env.current_player = 3 - env.current_player
        state = next_state
        total_reward += reward
        print('Total Reward:', total_reward)
        print('Current State:')
        env.render()
        if done:
            print('Game Over')
            if env.current_player == 2:
                print('first Win')
                first_win += 1
                break
            else:
                print('second Win')
                second_win += 1
                break
print(f"first win: {first_win}, first win rate: {first_win/episodes}  second win: {second_win}, second win rate: {second_win/episodes}")


    


print('Total Reward:', total_reward)