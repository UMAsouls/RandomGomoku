from N_Tuple import NTupleQAgent, NTupleMCTSAgent
from GomokuEnv import GomokuEnv
from NTupleGomokuEnv import NTupleGomokuEnv

from agent import RuleBasedAgent
from agent import RandomAgent
from agent import MinimaxAgent

import numpy as np

import time
import sys

BOARD_SIZE = 9  # ボードのサイズ
MODEL_DIR = "NTupleMCTSModel"
MODEL_NAME = "S9Model8_24_4"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

SIMULATION_TIME = 400
CPUCT = 1.0

def MCTSTurn(game: NTupleGomokuEnv, agent: NTupleMCTSAgent) -> bool:
    action, _ = agent.Search()
    set_act = (action % game.board_size, action // game.board_size) #x, yのタプルに変換

    _, _, done, _ = game.step(set_act)
    
    game.Animation()
    
    return done

def ReadStrBoard(game: NTupleGomokuEnv, s:str, side: str) -> np.ndarray:
    board = np.zeros((game.board_size, game.board_size), dtype=np.int64)
    
    for i in range(game.board_size):
        for j in range(game.board_size):
            v: int
            p = j + i*game.board_size
            if(s[p] == "-"): v = 0
            elif(s[p] == side): v = 1
            else: v = 2
            
            board[i,j] = v
            
    game.SetBoard(board)

def GetStrBoard(state):
    board_str = ""
    for i in state:
        for j in i:
            s: str
            if(j == 0): s = "-"
            elif(j == 1): s = "O"
            else: s = "X"
            
            board_str += s
            
    return board_str

def GetMove(game: NTupleGomokuEnv, c: int):
    (x,y) = (c % game.board_size, c // game.board_size)
    
    _, _, done, _ = game.step((x,y))
    
    return done

def SendMove(action: int):
    return f"move {action}"

def DoGo(game:NTupleGomokuEnv, t: int):
    action, _ = agent.Search()
    set_act = (action % game.board_size, action // game.board_size) #x, yのタプルに変換
    
    _, _, done, _ = game.step(set_act)
    
    return SendMove(action)


game = NTupleGomokuEnv(BOARD_SIZE,train_target="both")  # 先手で学習
agent = NTupleMCTSAgent(
    game, BOARD_SIZE, MODEL_PATH,
    cpuct = CPUCT, simulation_time=SIMULATION_TIME,
    epsilon=0
    )

agent.load()

game.reset()

board_str = GetStrBoard(game.GetBoard_Player1())

while True:
    cmd = sys.stdin.readline().strip().split()
    if cmd[0] == "quit":
        break
    
    if cmd[0] == "pos":
        ReadStrBoard(game, cmd[1], cmd[2])
    
    if cmd[0] == "move":
        GetMove(game, int(cmd[1]))
        
    if cmd[0] == "go":
        s = DoGo(game, int(cmd[1]))
        print(s)
        sys.stdout.flush()
        
    
    
