from RandomGomoku.Board.Board import Board
from RandomGomoku.Mass.HeadMass import HeadMass  # IHeadMassの実装例
from collections import defaultdict
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
if __name__ == "__main__":
    N = 100  # 生成する盤面数
    W, H = 9, 9  # 盤面サイズ
    stone_count = defaultdict(int)  # 座標ごとの設置回数
    for i in range(N):
        board = Board(HeadMass())
        board.MakeBoard(W, H)
        print(f"--- Board {i+1} ---")
        board.PrintBoard()
        # 盤面から石の座標を集計
        board_int = board.GetBoardInt()
        for y in range(H):
            for x in range(W):
                if board_int[y][x] != 0:
                    stone_count[(x, y)] += 1
        time.sleep(0.05)  # 見やすさのため少し待つ
    print("\n=== 石が置かれた座標の集計 ===")
    for y in range(H):
        row = []
        for x in range(W):
            row.append(f"{stone_count[(x, y)]:3}")
        print(" ".join(row))
