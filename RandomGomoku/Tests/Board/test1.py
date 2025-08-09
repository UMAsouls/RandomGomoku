
from RandomGomoku import IBoard, GetBoard

def Test():
    board: IBoard = GetBoard(15, 15)
    board.PrintBoard()