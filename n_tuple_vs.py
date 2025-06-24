from N_Tuple import NTupleQAgent
from GomokuEnv import GomokuEnv

from agent import RuleBasedAgent
from agent import RandomAgent
from agent import MinimaxAgent

import numpy as np

BOARD_SIZE = 15  # ボードのサイズ

game = GomokuEnv(BOARD_SIZE,train_target="first")  # 先手で学習
agent = NTupleQAgent(board_size=game.board_size)

rule_based_agent = RuleBasedAgent()
minimax_agent = MinimaxAgent()
random_agent = RandomAgent()

episodes = 100

train_agent = agent  # 学習エージェント
opponent_agent = rule_based_agent  # 対戦相手エージェント

for episode in range(episodes):
    game.reset()  # ゲームのリセット
    state = game.get_board()
    done = False
    total_reward = 0

    while not done:
        if(game.current_player == game.train_player):
            # 学習エージェントのターン
            action = train_agent.select_action(state)
            action = (action % game.board_size, action // game.board_size)
        else:
            # 対戦相手エージェントのターン
            action = opponent_agent.get_action(state, game.current_player)
            
        #print(set_act)
        next_state, reward, done, _ = game.step(action)
        
        next_state = np.array(next_state)  # 次の状態もNumPy配列

        if(game.current_player == game.train_player):
            # 学習エージェントのターンならば更新
            agent.update(state, action[1]*game.board_size + action[0], reward, next_state, done)
        state = next_state
        total_reward += reward
        #game.render()

    print(f"Episode: {episode}, Total Reward: {total_reward}")