from RandomGomoku import Board,FastBoard
from RandomGomoku import GetFastBoard

from RandomGomoku.Dependency import Dependency

import os

import sys

from RandomGomoku.const import Stone



def Main() -> None:
    board = GetFastBoard(15, 15)
    
    
    stone = Stone.BLACK
    s = "黒"
    while(True):
        board.PrintBoard()
        x = int(input(f"置くx座標({s}):"))
        if(x == -1): return
        y = int(input(f"置くy座標({s}):"))
        if(y == -1): return
        
        print("")
        if(board.SetStone(x,y,stone)): 
            board.PrintBoard()
            print(f"{s}の勝利!")
            return
            
        if(stone == Stone.WHITE):
            stone = Stone.BLACK
            s = "黒"
        else:
            stone = Stone.WHITE
            s = "白"



if __name__ == "__main__":
    Main()