from injector import inject
from random import randint

from Interfaces import IHeadMass, IMass

from const import Stone

from Board.RandomSetter import RandomSetter

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
        
        black_rand_size = (w*2//5, h*2//5)
        white_rand_size = (w*1//5, h*1//5)
        
        black_rand1: RandomSetter = RandomSetter(0, 0, black_rand_size[0], black_rand_size[1])
        black_rand2: RandomSetter = RandomSetter(w-black_rand_size[0], 0, black_rand_size[0], black_rand_size[1])
        black_rand3: RandomSetter = RandomSetter(0, h-black_rand_size[1], black_rand_size[0], black_rand_size[1])
        black_rand4: RandomSetter = RandomSetter(w-black_rand_size[0], h-black_rand_size[1], black_rand_size[0], black_rand_size[1])
        
        black_rands: list[RandomSetter] = [
            black_rand1,black_rand2,black_rand3,black_rand4
        ]
        
        white_rand1: RandomSetter = RandomSetter(black_rand_size[0], 0, white_rand_size[0], h)
        white_rand2: RandomSetter = RandomSetter(0, black_rand_size[1], w, white_rand_size[1])
        
        white_rands: list[RandomSetter] = [
            white_rand1, white_rand2
        ]
        
        for i in black_rands:
            pos = i.RandomMassGet()
            self.SetStone(pos[0], pos[1], Stone.BLACK)
            
        
        wpos1 = white_rand1.RandomMassGet()
        wpos2 = white_rand2.RandomMassGet()
        
        if(randint(0,1) == 0): wpos = wpos1
        else: wpos = wpos2
        
        self.SetStone(wpos[0], wpos[1], Stone.WHITE)
        
        
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
        sboard: str = ""
        chg_map: dict[int, str] = {0:"□", 1:"○", 2:"●"}
        
        for i in self.__board:
            for j in i:
                s = j.GetStatus()
                sboard += f" {chg_map[s]}"
            sboard += "\n"
            
        print(sboard)