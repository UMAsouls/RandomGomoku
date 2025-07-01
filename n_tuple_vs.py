from N_Tuple import NTupleQAgent
from GomokuEnv import GomokuEnv
from NTupleGomokuEnv import NTupleGomokuEnv

from agent import RuleBasedAgent
from agent import RandomAgent
from agent import MinimaxAgent

import numpy as np

import time

BOARD_SIZE = 15  # ボードのサイズ

game = NTupleGomokuEnv(BOARD_SIZE,train_target="first")  # 先手で学習
agent = NTupleQAgent(board_size=game.board_size, eps=0)

agent.load()

rule_based_agent = RuleBasedAgent(BOARD_SIZE)
minimax_agent = MinimaxAgent()
random_agent = RandomAgent(BOARD_SIZE)

opponent_agent = rule_based_agent   # 対戦相手エージェント



t0 = time.time()

vs = 10
win_n = 0
win_o = 0
for i in range(vs):
    game.reset()  # ゲームのリセット
    state = game.board.GetBoardInt()
    bef_state = np.nan
    bef_action = None
    done = False
    total_reward = 0

    select_time = 0
    game_step_time = 0
    update_time = 0

    step_num = 0
    while not done:
        action = agent.select_action(state)
        set_act = (action % game.board_size, action // game.board_size) #x, yのタプルに変換

        next_state, reward, done, _ = game.step(set_act)
        state = next_state

        total_reward += reward
        step_num += 1
        game.Animation()

        if done:
            win_n += 1
            break

        action = opponent_agent.get_action(state, game.current_player)

        next_state, reward, done, _ = game.step(action)
        state = next_state

        total_reward += reward
        step_num += 1
        game.Animation()

        if done:
            win_o += 1
            break

    game.AnimationEnd()
    game.render()

print(f"n_tupleの勝率:{win_n/vs * 100}%, opposeの勝率:{win_o/vs * 100}%")



    
    
    