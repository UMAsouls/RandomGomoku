from RandomGomoku import GetBoard
from RandomGomoku import Board

from RandomGomoku.const import Stone

import random

from typing import Callable

WIDTH = 19
HEIGHT = 19

def Agent_Random(board: Board, stone: Stone) -> bool:
    b = board.GetBoardInt()
    
    x = random.randint(0,WIDTH-1)
    y = random.randint(0,HEIGHT-1)
    
    while(b[y][x] != 0):
        x = random.randint(0,WIDTH-1)
        y = random.randint(0,HEIGHT-1)
    
    return board.SetStone(x,y,stone)
    

def Main(agent_black: Callable[[Board, Stone], bool], agent_white: Callable[[Board, Stone], bool]) -> None:
    board: Board = GetBoard(WIDTH,HEIGHT)
    
    while(True):
        board.PrintBoard()
        if(agent_black(board, Stone.BLACK)): 
            board.PrintBoard()
            print(f"黒の勝利!")
            return
            
        board.PrintBoard()
        if(agent_white(board, Stone.WHITE)): 
            board.PrintBoard()
            print(f"白の勝利!")
            return
            
Main(Agent_Random, Agent_Random)