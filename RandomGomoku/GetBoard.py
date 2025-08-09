from RandomGomoku.Interfaces import IBoard
from RandomGomoku.Board import FastBoard

from RandomGomoku.Dependency import Dependency

import os

from RandomGomoku.const import Stone



def GetBoard(w: int, h: int) -> IBoard:
    board: IBoard = FastBoard()
    board.MakeBoard(w,h)
    
    return board
