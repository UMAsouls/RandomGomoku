from Board import Board

from Dependency import Dependency

import os

from const import Stone

def GetBoard(w: int, h: int) -> Board:
    container = Dependency()
    board: Board = container.resolve(Board)
    board.MakeBoard(w,h)
    
    return board