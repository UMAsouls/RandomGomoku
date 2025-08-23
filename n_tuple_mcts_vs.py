from N_Tuple import NTupleQAgent, NTupleMCTSAgent
from GomokuEnv import GomokuEnv
from NTupleGomokuEnv import NTupleGomokuEnv

from agent import RuleBasedAgent
from agent import RandomAgent
from agent import MinimaxAgent

import numpy as np

import time

BOARD_SIZE = 9  # ボードのサイズ
MODEL_DIR = "NTupleMCTSModel"
MODEL_NAME = "S9Model8_23_2"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

SIMULATION_TIME = 400
CPUCT = 1.0

game = NTupleGomokuEnv(BOARD_SIZE,train_target="both")  # 先手で学習
agent = NTupleMCTSAgent(
    game, BOARD_SIZE, MODEL_PATH,
    cpuct = CPUCT, simulation_time=SIMULATION_TIME,
    epsilon=0
    )

agent.load()

rule_based_agent = RuleBasedAgent(BOARD_SIZE)
minimax_agent = MinimaxAgent(board_size=BOARD_SIZE)
random_agent = RandomAgent(BOARD_SIZE)

opponent_agent = minimax_agent   # 対戦相手エージェント



t0 = time.time()

vs = 100
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
        action, _ = agent.Search()
        set_act = (action % game.board_size, action // game.board_size) #x, yのタプルに変換

        next_state, reward, done, _ = game.step(set_act)
        state = next_state

        total_reward += reward
        step_num += 1
        game.Animation()

        if done:
            win_n += 1
            break

        action = opponent_agent.get_action(state, 1)

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