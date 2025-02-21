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
        else:
            raise ValueError("train_targetはfirstかsecondを指定してください")


    def step(self, action):
        x, y = action
        if self.board.GetBoardInt()[y][x] != 0:
            raise ValueError("無効なアクション : 既に埋まっているセル")
        
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
            if self.current_player == self.train_player:
                reward += 10000  # 黒が勝った
            else:
                reward += -10000  # 白が勝った

        # ゲームが終了しない場合（続行）
        else:
            # 現在の手の進行具合を考慮して報酬を調整
            reward += self.evaluate_action(x, y)

            # 石を置いた位置に基づく報酬（戦略的な位置）
            strategic_reward = self.evaluate_strategic_position(x, y)
            reward += strategic_reward


        
        return self.board.copy(), reward, done, {}

    def evaluate_action(self, x, y):
        """
        評価関数: 現在の手を評価して報酬を返す
        x, y: 石を置いた位置
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 横、縦、斜めの4方向
        total_score = 0
        for dx, dy in directions:
            score = self.evaluate_direction(x, y, dx, dy)
            total_score += score

        return total_score

    def evaluate_direction(self, x, y, dx, dy):
        """
        一方向の評価関数
        x, y: 石を置いた位置
        dx, dy: 評価する方向（横、縦、斜め）
        """
        # 連続する石と空きマス、相手の石を数える
        stone_count = 1
        empty_count = 0
        opponent_stone_count = 0
        opponent_stone_blocked = 0  # 相手の石を防ぐためのカウント

        # 1方向に延びている石を数える
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board.GetBoardInt()[ny][nx] == self.train_player:
                    stone_count += 1
                elif self.board.GetBoardInt()[ny][nx] == 0:
                    empty_count += 1
                else:
                    opponent_stone_count += 1
                    break
            else:
                break

        # 反対方向にも評価
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board.GetBoardInt()[ny][nx] == self.train_player:
                    stone_count += 1
                elif self.board.GetBoardInt()[ny][nx] == 0:
                    empty_count += 1
                else:
                    opponent_stone_count += 1
                    break
            else:
                break

        # 攻撃パターン（5連続の石や4連続の石を作る）
        if stone_count == 5:
            return 10000  # 完全に勝ちを確定させる場合
        if stone_count == 4 and empty_count == 1:
            return 5000  # 4連続を作れる場合
        if stone_count == 3 and empty_count == 2:
            return 1000  # 3連続を作れる場合
        if stone_count == 2 and empty_count == 3:
            return 500

        # 相手の妨害（相手が4つ並べていたら阻止）
        if opponent_stone_count == 1 and empty_count == 1:
            return -10000  # 相手の4連続を防ぐ
        if opponent_stone_count == 2 and empty_count == 1:
            return -5000  # 相手の3連続を防ぐ

        # 空きスペースの多い連続石は価値が高い
        score = 0
        if empty_count > 0:
            score += 10 * empty_count

        return score

    def evaluate_strategic_position(self, x, y):
        """
        戦略的な位置に置かれた石に対する報酬
        中央に近いほど高い価値をつける
        """
        center_x, center_y = self.board_size // 2, self.board_size // 2
        distance = abs(center_x - x) + abs(center_y - y)  # 中心からの距離を計算

        # 中央に近いほど高い報酬
        strategic_score = max(0, 100 - distance)  # 中心からの距離が近いほど報酬が高くなる
        return strategic_score


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
