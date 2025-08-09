
from RandomGomoku import IBoard, GetBoard

def Test():
    board: IBoard = GetBoard(9, 9)
    board.PrintBoard()