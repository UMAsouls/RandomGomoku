from N_Tuple import NTupleQAgent, NTupleMCTSAgent
from GomokuEnv import GomokuEnv
from NTupleGomokuEnv import NTupleGomokuEnv

from agent import RuleBasedAgent
from agent import RandomAgent
from agent import MinimaxAgent

import numpy as np

import time

BOARD_SIZE = 9  # ボードのサイズ
MODEL_DIR = "NTupleMCTSModel2"
MODEL_NAME = "S9Model8_26_2"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

SIMULATION_TIME = 400
CPUCT = 0.01
USE_V = True

def MCTSTurn(game: NTupleGomokuEnv, agent: NTupleMCTSAgent) -> bool:
    action, _ = agent.Search()
    set_act = (action % game.board_size, action // game.board_size) #x, yのタプルに変換

    _, _, done, _ = game.step(set_act)
    
    game.Animation()
    
    return done

def OpponentTurn(game: NTupleGomokuEnv, agent: RuleBasedAgent) -> bool:
    action = agent.get_action(game.GetBoard_CurrentPlayer(), 1)

    _, _, done, _ = game.step(action)
    
    game.Animation()
    
    return done


game = NTupleGomokuEnv(BOARD_SIZE,train_target="both")  # 先手で学習
agent = NTupleMCTSAgent(
    game, BOARD_SIZE, MODEL_PATH,
    cpuct = CPUCT, simulation_time=SIMULATION_TIME,
    epsilon=0, use_v=USE_V
    )

agent.load()

rule_based_agent = RuleBasedAgent(BOARD_SIZE)
minimax_agent = MinimaxAgent(board_size=BOARD_SIZE)
random_agent = RandomAgent(BOARD_SIZE)

opponent_agent = rule_based_agent   # 対戦相手エージェント



t0 = time.time()

vs = 100
win_n = 0
win_o = 0
player1_is_mcts = True
for i in range(vs):
    game.reset()  # ゲームのリセット
    state = game.board.GetBoardInt()
    done = False

    while not done:
        if(player1_is_mcts):
            if(game.current_player == 1):
                done = MCTSTurn(game, agent)
                win_n += done
            else:
                done = OpponentTurn(game,opponent_agent)
                win_o += done
        else:
            if(game.current_player == 2):
                done = MCTSTurn(game, agent)
                win_n += done
            else:
                done = OpponentTurn(game,opponent_agent)
                win_o += done

    game.AnimationEnd()
    game.render()
    
    player1_is_mcts = not player1_is_mcts

print(f"n_tupleの勝率:{win_n/vs * 100}%, opposeの勝率:{win_o/vs * 100}%")