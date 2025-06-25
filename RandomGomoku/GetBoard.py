from RandomGomoku.Board import Board
from RandomGomoku.Board import FastBoard

from RandomGomoku.Dependency import Dependency

import os

from RandomGomoku.const import Stone

def GetBoard(w: int, h: int) -> Board:
    container = Dependency()
    board: Board = container.resolve(Board)
    board.MakeBoard(w,h)
    
    return board

def GetFastBoard(w: int, h: int) -> FastBoard:
    board: FastBoard = FastBoard()
    board.MakeBoard(w,h)
    return board