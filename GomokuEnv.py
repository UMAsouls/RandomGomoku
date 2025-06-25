import numpy as np
from RandomGomoku.Board import Board
from RandomGomoku.Dependency import Dependency
from RandomGomoku.const import Stone

import numpy as np
from RandomGomoku.const import Stone

class GomokuEnv:
    def __init__(self, board_size=19,train_target = "first"):
        self.board_size = board_size
        self.container = Dependency()
        self.board: Board = self.container.resolve(Board)
        self.board.MakeBoard(board_size, board_size)
        self.stone = Stone.BLACK
        self.current_player = 1
        self.blackStones = 0
        self.whiteStones = 0
        if train_target == "first":
            self.train_player = 1
        elif train_target == "second":
            self.train_player = 2
        elif train_target == "both":
            self.train_player = 0
        else:
            raise ValueError("train_targetはfirstかsecondを指定してください")
        

    def reset(self):
        self.board.MakeBoard(self.board_size, self.board_size)
        self.stone = Stone.BLACK
        self.current_player = 1
        self.blackStones = 0
        self.whiteStones = 0


    def step(self, action):
        x, y = action
        if self.board.GetBoardInt()[y][x] != 0:
            raise ValueError(f"無効なアクション : 既に埋まっているセル:[{x}, {y}]")
        
        if self.current_player == 1:
            self.stone = Stone.BLACK
            self.blackStones += 1
            
        else:
            self.stone = Stone.WHITE
            self.whiteStones += 1
            

        done = self.board.SetStone(x, y, self.stone)
        
        #石の数が正常かチェック
        if not(self.blackStones-self.whiteStones == 1 or self.blackStones == self.whiteStones):
            print(self.blackStones)
            print(self.whiteStones)
            raise ValueError("石の数がおかしいです")

        # 報酬の初期設定
        reward = 0

        # ゲームが終了した場合
        if done:
            self.board.PrintBoard()
            if self.current_player == self.train_player or self.train_player == 0:
                reward += 1  # 黒が勝った
            else:
                reward += -1  # 白が勝った
        else:
            reward += 0

        # 次のプレイヤーに交代
        self.current_player = 3 - self.current_player
        
        return self.board.GetBoardInt(), reward, done, {}


    def get_board(self):
        if self.current_player != 1 and self.train_player == 0:
            # 白の視点でボードを返す
            inverted_board = np.zeros_like(self.board.GetBoardInt())
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if self.board.GetBoardInt()[y][x] == 1:
                        inverted_board[y][x] = 2
                    elif self.board.GetBoardInt()[y][x] == 2:
                        inverted_board[y][x] = 1
            return inverted_board
        else:
            # 黒の視点でボードを返す
            return np.array(self.board.GetBoardInt())

    def get_human_action(self):
        while True:
            x = int(input("x座標を入力してください: "))
            y = int(input("y座標を入力してください: "))
            if self.board.GetBoardInt()[y][x] == 0:
                break
            else:
                print("そこには置けません")
        return (y, x)

    def render(self):
        self.board.PrintBoard()