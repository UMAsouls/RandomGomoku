from injector import inject
from random import randint

from RandomGomoku.Interfaces import IHeadMass, IMass, IBoard

from RandomGomoku.const import Stone

from RandomGomoku.Board.RandomSetter import RandomSetter

import numpy as np

BLACKINT = 1
WHITEINT = 2

class FastBoard(IBoard):
    def __init__(self) -> None:
        self.__board: np.ndarray
        
    def AltRandomSet(self):
        w = self.__board.shape[1]
        h = self.__board.shape[0]

        # --- 白石（青）の配置 ---
        # 4つの区画の基点となる座標を定義
        # (例: 8x8の場合、(2,2), (6,2), (2,6), (6,6) を中心としたエリア)
        quadrant_origins = [(w//4-1, h//4-1), (w - w//4-1, h//4-1), 
                            (w//4-1, h - h//4-1), (w - w//4-1, h - h//4-1)]

        # 各区画内でランダムな座標を決めるための範囲 (2x2)
        random_area_size = (2, 2)
        
        # 各区画のRandomSetterを作成
        random_setters: list[RandomSetter] = []
        for origin in quadrant_origins:
            # 2x2の範囲を持つRandomSetterをリストに追加
            random_setters.append(RandomSetter(origin[0], origin[1], 
                                            random_area_size[0], random_area_size[1]))
        
        # 各区画に1つずつ石を置く
        for setter in random_setters:
            pos = setter.RandomMassGet()
            
            # 取得した座標に白石を置く
            self.SetStone(pos[0], pos[1], Stone.WHITE)
            
        # --- 黒石（赤）の配置 ---
        # 中央の座標を計算
        center_pos = (w//2, h//2)
        
        # 中央に黒石を置く
        self.SetStone(center_pos[0], center_pos[1], Stone.BLACK)
    
    def RandomSet(self):
        w = self.__board.shape[1]
        h = self.__board.shape[0]
        
        white_rand_size = (w*2//5, h*2//5)
        black_rand_size = (w*1//5, h*1//5)
        
        white_rand1: RandomSetter = RandomSetter(0, 0, white_rand_size[0], white_rand_size[1])
        white_rand2: RandomSetter = RandomSetter(w-white_rand_size[0], 0, white_rand_size[0], white_rand_size[1])
        white_rand3: RandomSetter = RandomSetter(0, h-white_rand_size[1], white_rand_size[0], white_rand_size[1])
        white_rand4: RandomSetter = RandomSetter(w-white_rand_size[0], h-white_rand_size[1], white_rand_size[0], white_rand_size[1])
        
        white_rands: list[RandomSetter] = [
            white_rand1,white_rand2,white_rand3,white_rand4
        ]
        
        black_rand1: RandomSetter = RandomSetter(white_rand_size[0], 0, black_rand_size[0], h)
        black_rand2: RandomSetter = RandomSetter(0, white_rand_size[1], w, black_rand_size[1])
        
        black_rands: list[RandomSetter] = [
            black_rand1, black_rand2
        ]
        
        for i in white_rands:
            pos = i.RandomMassGet()
            self.SetStone(pos[0], pos[1], Stone.WHITE)
            
        
        bpos1 = black_rand1.RandomMassGet()
        bpos2 = black_rand2.RandomMassGet()
        
        if(randint(0,1) == 0): bpos = bpos1
        else: bpos = bpos2
        
        self.SetStone(bpos[0], bpos[1], Stone.BLACK)
    
    def MakeBoard(self, w:int, h:int):
        self.__board: np.ndarray = np.zeros((w,h), dtype=np.int32)
        
        self.__board_oppose = np.zeros((w,h), dtype=np.int32)
        
        self.__board_wb = [self.__board, self.__board_oppose]
        
        #self.RandomSet()
        
        #ルール変更に対応
        self.AltRandomSet()
        
    def SetBoard(self, board:np.ndarray) -> None:
        self.__board = board
        
    def GetBoardInt(self) -> np.ndarray:
        return self.__board.copy()
    
    def GetBoardOppose(self) -> np.ndarray:
        return self.__board_oppose.copy()
    
    def GetStatus(self, x: int, y: int) -> int:
        return self.__board[y][x]
    
    def SetStone(self, x: int, y: int, stone: Stone) -> bool:
        self.__board[y][x] = BLACKINT if stone == Stone.BLACK else WHITEINT
        self.__board_oppose[y][x] = WHITEINT if stone == Stone.BLACK else BLACKINT
        
        player = BLACKINT if stone == Stone.BLACK else WHITEINT
        
        return self.JudgeWin(x, y, player)
        
    def JudgeWin(self, x: int, y: int, player: int) -> int:
        if(player != WHITEINT and player != BLACKINT):
            raise ValueError(f"Player must be {WHITEINT} or {BLACKINT}")
        
        # Check horizontal
        horizon_count = self.CountByDir(x, y, 1, 0, player) + self.CountByDir(x, y, -1, 0, player) - 1
        #print(f"horizon: {horizon_count}")
        if horizon_count >= 5:
            return 1
        
        # Check vertical
        vertical_count = self.CountByDir(x, y, 0, 1, player) + self.CountByDir(x, y, 0, -1, player) - 1
        #print(f"vertical: {vertical_count}")
        if vertical_count >= 5:
            return 1
        
        # Check diagonal /
        diagonal1_count = self.CountByDir(x, y, 1, -1, player) + self.CountByDir(x, y, -1, 1, player) - 1
        #print(f"diagonal /: {diagonal1_count}")
        if diagonal1_count >= 5:
            return 1
        
        # Check diagonal \
        diagonal2_count = self.CountByDir(x, y, 1, 1, player) + self.CountByDir(x, y, -1, -1, player) - 1
        #print(f"diagonal \\ : {diagonal2_count}")
        if diagonal2_count >= 5:
            return 1
        
        if(len(np.where(self.__board == 0)[0]) == 0):
            return 2
        
        return 0
        
    def CountByDir(self, x: int, y: int, dx: int, dy: int, player: int) -> int:
        px, py = x, y
        count = 0
        while 0 <= px < self.__board.shape[1] and 0 <= py < self.__board.shape[0]:
            if self.__board_wb[player-1][py,px] == 1:
                count += 1
            else:
                break
            px += dx
            py += dy
            
        return count
    
    def StringBoard(self) -> str:
        count1: int = 0
        count2: int = 0
        sboard: str = ""
        chg_map: dict[int, str] = {0:"🔳", 1:"🔴", 2:"🔵"}
        for i in self.__board:
            for j in i:
                sboard += f"{chg_map[j]}"
                if(j == 1): count1 += 1
                elif(j == 2): count2 += 1
            sboard += "\n"
        sboard += f"\nBlack: {count1} White: {count2}"
        
        return sboard
    
    def PrintBoard(self) -> None:
        print(self.StringBoard())

class Board():
    
    @inject
    def __init__(self, headmass: IHeadMass) -> None:
        self.__headmass: IHeadMass = headmass
        self.__board: list[list[IMass]] = []
        
        self.__width: int = -1
        self.__height: int = -1
        
        
    
    def SetStone(self, x: int, y: int, stone: Stone) -> bool:
        return self.__board[y][x].SetStone(stone)
    
    def RandomSet(self):
        w:int = self.__width
        h: int = self.__height
        
        white_rand_size = (w*2//5, h*2//5)
        black_rand_size = (w*1//5, h*1//5)
        
        white_rand1: RandomSetter = RandomSetter(0, 0, white_rand_size[0], white_rand_size[1])
        white_rand2: RandomSetter = RandomSetter(w-white_rand_size[0], 0, white_rand_size[0], white_rand_size[1])
        white_rand3: RandomSetter = RandomSetter(0, h-white_rand_size[1], white_rand_size[0], white_rand_size[1])
        white_rand4: RandomSetter = RandomSetter(w-white_rand_size[0], h-white_rand_size[1], white_rand_size[0], white_rand_size[1])
        
        white_rands: list[RandomSetter] = [
            white_rand1,white_rand2,white_rand3,white_rand4
        ]
        
        black_rand1: RandomSetter = RandomSetter(black_rand_size[0], 0, black_rand_size[0], h)
        black_rand2: RandomSetter = RandomSetter(0, black_rand_size[1], w, black_rand_size[1])
        
        black_rands: list[RandomSetter] = [
            black_rand1, black_rand2
        ]
        
        for i in white_rands:
            pos = i.RandomMassGet()
            self.SetStone(pos[0], pos[1], Stone.WHITE)
            
        
        bpos1 = black_rand1.RandomMassGet()
        bpos2 = black_rand2.RandomMassGet()
        
        if(randint(0,1) == 0): bpos = bpos1
        else: bpos = bpos2
        
        self.SetStone(bpos[0], bpos[1], Stone.BLACK)
        
        
    def MakeBoard(self, w:int, h:int):
        self.__board = self.__headmass.MakeBoard(w,h)
        
        self.__width = w
        self.__height = h
        
        self.RandomSet()
        
        
    def GetBoardInt(self) -> list[list[int]]:
        return [
            [
                i.GetStatus() for i in j
            ]
            for j in self.__board
        ]
        
        
    def PrintBoard(self) -> None:
        count1: int = 0
        count2: int = 0
        sboard: str = ""
        chg_map: dict[int, str] = {0:"🔳", 1:"🔴", 2:"🔵"}
        for i in self.__board:
            for j in i:
                s = j.GetStatus()
                sboard += f"{chg_map[s]}"
                if(s == 1): count1 += 1
                elif(s == 2): count2 += 1
            sboard += "\n"
            
        print(sboard)
        print(f"Black: {count1} White: {count2}")
  
    def copy(self): 
        return self.GetBoardInt()