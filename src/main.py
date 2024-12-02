from Board import Board

from Dependency import Dependency

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def Main() -> None:
    container = Dependency()
    board: Board = container.resolve(Board)
    board.MakeBoard(5,5)
    board.PrintBoard()


if __name__ == "__main__":
    Main()
    print("hello")