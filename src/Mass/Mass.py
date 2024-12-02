
from collections.abc import Iterator
from Interfaces import IMass, CreatingMass
from const import Stone


class Mass(IMass,CreatingMass):
    
    
    def __init__(self) -> None:
        
        #方向に合わせたマス
        self.__topleft :Mass = None
        self.__top :Mass = None
        self.__topright :Mass = None
        self.__left :Mass = None
        self.__right :Mass = None
        self.__bottomleft :Mass = None
        self.__bottom :Mass = None
        self.__bottomright :Mass = None
        
        
        self.__around :list[list[Mass]] = [
            [self.__topleft, self.__top, self.__topright],
            [self.__left, None, self.__right],
            [self.__bottomleft, self.__bottom, self.__bottomright]
        ]
        """マスをidxで取り出しやすくしたもの
        """
        
        
        self.stone :Stone = Stone.NONE
        
        
    @property
    def Topleft(self) -> IMass:
        return self.__topleft
    
    @property
    def Top(self) -> IMass:
        return self.__top
    
    @property
    def Topright(self) -> IMass:
        return self.__topright
    
    @property
    def Left(self) -> IMass:
        return self.__left
    
    @property
    def Right(self) -> IMass:
        return self.__right
    
    @property
    def Bottomleft(self) -> IMass:
        return self.__bottomleft
    
    @property
    def Bottom(self) -> IMass:
        return self.__bottom
    
    @property
    def Bottomright(self) -> IMass:
        return self.__bottomright
    
    @Right.setter
    def Right(self, v: CreatingMass) -> None:
        self.__right = v
        
    @Bottom.setter
    def Bottom(self, v:CreatingMass) -> None:
        self.__bottom = v
        
        
    
    def AddAccessor(self) -> None:
        
        if(self.__bottom == None and self.__right == None): return
        
        if(self.__bottom == None):
            self.__right.__left = self
            self.__right.AddAccessor()
            return
            
        self.__bottom.__top = self
        
        if(self.__right == None): return
        
        self.__right.__left = self
        self.__bottomright = self.__bottom.__right
        
        self.__bottomright.__topleft = self
        self.__right.__bottomleft = self.__bottom
        self.__bottom.__topright = self.__right
        
        self.__right.AddAccessor()
        if(self.__left == None): self.__bottom.AddAccessor()
        
        
    def SetStone(self, stone: Stone) -> bool:
        self.stone = stone
        
        ans1 = self.Count((0,0),0,self.stone) + self.Count((2,2),0,self.stone) >= 5
        ans2 = self.Count((0,1),0,self.stone) + self.Count((2,1),0,self.stone) >= 5
        ans3 = self.Count((0,2),0,self.stone) + self.Count((2,0),0,self.stone) >= 5
        
        return ans1 or ans2 or ans3
        
        
        
        
    def Count(self, dir: tuple[int,int], count:int, kind: Stone) -> int:
        if(kind != self.stone or self.__around[dir[1]][dir[0]] == None): 
            return count
        
        return self.__around[dir[1]][dir[0]].Count(dir,count+1,kind)
        
        
    def GetStatus(self) -> int:
        if(self.stone == Stone.BLACK): return 1
        elif(self.stone == Stone.WHITE): return 2
        else: return 0
    
        
        
        
        
        
        
    
    
        