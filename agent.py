from RandomGomoku import GetBoard
from RandomGomoku import Board

from RandomGomoku.const import Stone

import random

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
    

def Main() -> None:
    board: Board = GetBoard(WIDTH,HEIGHT)
    
    
    stone = Stone.BLACK
    s = "黒"
    while(True):
        board.PrintBoard()
        if(Agent_Random(board, stone)): 
            board.PrintBoard()
            print(f"{s}の勝利!")
            return
            
        if(stone == Stone.WHITE):
            stone = Stone.BLACK
            s = "黒"
        else:
            stone = Stone.WHITE
            s = "白"
            
Main()