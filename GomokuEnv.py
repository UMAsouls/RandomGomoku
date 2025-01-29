import numpy as np
from RandomGomoku.Board import Board
from RandomGomoku.Dependency import Dependency
from RandomGomoku.const import Stone




class GomokuEnv:
    def __init__(self, board_size=19, stone=Stone.BLACK):
        self.board_size = board_size
        self.container = Dependency()
        self.board: Board = self.container.resolve(Board)
        self.board.MakeBoard(board_size, board_size)
        self.stone = stone
        self.current_player = 1
        self.blackStones = 0
        self.whiteStones = 0
        self.reset()

    def reset(self):
        self.container = Dependency()
        self.board: Board = self.container.resolve(Board)
        self.board.MakeBoard(self.board_size,self.board_size)
        self.current_player = 1
        self.blackStones = 0
        self.whiteStones = 0
        return self.board.copy()

    def step(self, action):
        x, y = action
        # print(self.board.GetBoardInt())
        if self.board.GetBoardInt()[y][x] != 0:
            # print(y,x)
            raise ValueError("無効なアクション : 既に埋まっているセル")
        if self.current_player == 1:
            self.stone = Stone.BLACK
        else:
            self.stone = Stone.WHITE
        # print(f"Player {self.current_player} plays {self.stone} at ({x}, {y})")
        done = self.board.SetStone(x, y, self.stone)
        if self.stone == Stone.BLACK:
            self.blackStones += 1
        else:
            self.whiteStones += 1
            
        if not(self.blackStones-self.whiteStones == 1 or self.blackStones == self.whiteStones):
            raise ValueError("石の数がおかしいです")
        
        # self.board.PrintBoard()

        if done:
            reward = 1 if self.current_player == 1 else -1
        else:
            reward = 0
        
        # 次のプレイヤーに交代
        self.current_player = 3 - self.current_player  # 交代後、次のプレイヤーに切り替え

        return self.board.copy(), reward, done, {}
