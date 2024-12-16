from RandomGomoku.Board import Board

from RandomGomoku.Dependency import Dependency

import os

import sys

from RandomGomoku.const import Stone



def Main() -> None:
    container = Dependency()
    board: Board = container.resolve(Board)
    board.MakeBoard(19,19)
    
    
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