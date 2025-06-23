from collections.abc import Iterator

from RandomGomoku.Interfaces import IHeadMass, CreatingMass

from RandomGomoku.Mass import Mass

from RandomGomoku.const import Stone

class HeadMass(IHeadMass, Mass):
    def __init__(self) -> None:
        super().__init__(0,0)
        
        self.width: int = 0
        self.height: int = 0
        
    def MakeBoard(self, width: int, height: int) -> list[list[CreatingMass]]:
        
        self.width = width
        self.height = height
        
        self.stone = Stone.NONE
        
        board: list[list[CreatingMass]] = [
            [ 
             Mass(i,j) for i in range(width)
            ]
            for j in range(height)
        ]
        
        board[0][0] = self
        
        for i in range(height):
            for j in range(width-1):
                board[i][j].Right = board[i][j+1]
                
        for i in range(height-1):
            for j in range(width):
                board[i][j].Bottom = board[i+1][j]
                
                
        self.AddAccessor()
        
        return board
    
    def AccessorTest(self) -> bool:
        
        mass = self
        for i in range(self.height):
            left = mass
            for j in range(self.width):
                if(mass.x != j or mass.y != i):
                    return False
                
                if(mass.Right is not None):
                    if(mass.Right.x != j+1 or mass.Right.y != i):
                        return False
                    
                    
                if(mass.Bottom is not None):
                    if(mass.Bottom.x != j or mass.Bottom.y != i+1):
                        return False
                    
                if(mass.Left is not None):
                    if(mass.Left.x != j-1 or mass.Left.y != i):
                        return False
                    
                if(mass.Top is not None):
                    if(mass.Top.x != j or mass.Top.y != i-1):
                        return False
                
                if(mass.Topleft is not None):
                    if(mass.Topleft.x != j-1 or mass.Topleft.y != i-1):
                        return False
                    
                if(mass.Topright is not None):
                    if(mass.Topright.x != j+1 or mass.Topright.y != i-1):
                        return False
                
                if(mass.Bottomleft is not None):
                    if(mass.Bottomleft.x != j-1 or mass.Bottomleft.y != i+1):
                        return False
                
                if(mass.Bottomright is not None):
                    if(mass.Bottomright.x != j+1 or mass.Bottomright.y != i+1):
                        return False
                
                
                mass = mass.Right
            mass = left.Bottom
            
        return True