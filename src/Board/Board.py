from injector import inject

from Interfaces import IHeadMass, IMass

from const import Stone

class Board():
    
    @inject
    def __init__(self, headmass: IHeadMass) -> None:
        self.__headmass: IHeadMass = headmass
        self.__board: list[list[IMass]] = []
        
        self.__width: int = -1
        self.__height: int = -1
        
    
    def SetStone(self, x: int, y: int, stone: Stone) -> bool:
        return self.__board[y][x].SetStone(stone)
        
        
    def MakeBoard(self, w:int, h:int):
        self.__board = self.__headmass.MakeBoard(w,h)
        
        self.__width = w
        self.__height = h
        
    def GetBoardInt(self) -> list[list[int]]:
        return [
            [
                i.GetStatus() for i in j
            ]
            for j in self.__board
        ]
        
    def PrintBoard(self) -> None:
        sboard: str = ""
        
        for i in self.__board:
            for j in i:
                sboard += f" {j.GetStatus()}"
            sboard += "\n"
            
        print(sboard)