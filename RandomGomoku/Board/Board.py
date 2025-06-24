from injector import inject
from random import randint

from RandomGomoku.Interfaces import IHeadMass, IMass

from RandomGomoku.const import Stone

from RandomGomoku.Board.RandomSetter import RandomSetter

import numpy as np

class FastBoard:
    def __init__(self, board: list[list[int]]) -> None:
        self.__board: np.ndarray = np.array(board, dtype=np.int32)
        self.__board_white: np.ndarray = np.where(self.__board == Stone.WHITE, 1, 0)
        self.__board_black: np.ndarray = np.where(self.__board == Stone.BLACK, 1, 0)
        
    def GetBoard(self) -> list[list[int]]:
        return self.__board
    
    def GetStatus(self, x: int, y: int) -> int:
        return self.__board[y][x]
    
    def GetStatusFromWhite(self, x: int, y: int) -> int:
        if self.__board_white[y][x] == 1:
            return 1
        elif self.__board_black[y][x] == 1:
            return 2
        else:
            return 0
        
    def GetStatusFromBlack(self, x: int, y: int) -> int:
        if self.__board_black[y][x] == 1:
            return 1
        elif self.__board_white[y][x] == 1:
            return 2
        else:
            return 0
        
    def GetStatusFrom(self, x: int, y: int, player: int) -> int:
        if player == Stone.WHITE:
            return self.GetStatusFromWhite(x, y)
        elif player == Stone.BLACK:
            return self.GetStatusFromBlack(x, y)
        else:
            raise ValueError("Invalid player type. Use Stone.WHITE or Stone.BLACK.")
    
    def SetStatus(self, x: int, y: int, status: int) -> None:
        self.__board[y][x] = status

class Board():
    
    @inject
    def __init__(self, headmass: IHeadMass) -> None:
        self.__headmass: IHeadMass = headmass
        self.__board: list[list[IMass]] = []
        self.__board_int: FastBoard = FastBoard([])
        
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

        self.__board_int = FastBoard(self.GetBoardInt())
        
        
    def GetBoardInt(self) -> list[list[int]]:
        return [
            [
                i.GetStatus() for i in j
            ]
            for j in self.__board
        ]
    
    def GetFastBoard(self) -> FastBoard:
        return self.__board_int
        
        
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