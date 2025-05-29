import torch
import numpy as np
from RandomGomoku.Board import Board
from RandomGomoku.Dependency import Dependency
from RandomGomoku.const import Stone

class GomokuEnv:
    def __init__(self, board_size=19, train_target="first", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.board_size = board_size
        self.container = Dependency()
        self.board: Board = self.container.resolve(Board)
        self.board.MakeBoard(board_size, board_size)
        self.stone = Stone.BLACK
        self.current_player = 1
        self.blackStones = 0
        self.whiteStones = 0
        self.device = device
        if train_target == "first":
            self.train_player = 1
        elif train_target == "second":
            self.train_player = 2
        else:
            raise ValueError("train_targetはfirstかsecondを指定してください")

    def step(self, action):
        # Noneアクションのチェック
        if action is None:
            # 無効なアクション：Noneが渡された
            reward = torch.tensor(0.0, device=self.device)
            done = True
            board_tensor = torch.tensor(self.board.GetBoardInt(), dtype=torch.float32, device=self.device)
            return board_tensor, reward, done, {"invalid_action": True}
            
        x, y = action
        if self.board.GetBoardInt()[y][x] != 0:
            # 無効なアクション：既に埋まっているセルが選択された
            # これは盤面がすべて埋まった場合などに発生する可能性がある
            # エラーを投げる代わりに引き分けとして扱う
            reward = torch.tensor(0.0, device=self.device)
            done = True
            board_tensor = torch.tensor(self.board.GetBoardInt(), dtype=torch.float32, device=self.device)
            return board_tensor, reward, done, {"invalid_action": True}

        if self.current_player == 1:
            self.stone = Stone.BLACK
            self.blackStones += 1
        else:
            self.stone = Stone.WHITE
            self.whiteStones += 1

        done = self.board.SetStone(x, y, self.stone)

        # 石の数が正常かチェック
        if not (self.blackStones - self.whiteStones == 1 or self.blackStones == self.whiteStones):
            print(self.blackStones)
            print(self.whiteStones)
            raise ValueError("石の数がおかしいです")

        # 報酬の初期設定
        reward = torch.tensor(0.0, device=self.device)

        # ゲームが終了した場合
        if done:
            self.board.PrintBoard()
            if self.current_player == self.train_player:
                reward += 1.0  # 黒が勝った
            else:
                reward += -1.0  # 白が勝った
        else:
            reward += 0.0

        # 現プレイヤーを変数に保存
        now_player = self.current_player

        # 次のプレイヤーに交代
        self.current_player = 3 - self.current_player

        # 盤面をPyTorchテンソルに変換してGPUに転送
        board_tensor = torch.tensor(self.board.GetBoardInt(), dtype=torch.float32, device=self.device)
        return board_tensor, reward, done, {"which_player":now_player}

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