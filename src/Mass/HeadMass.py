from collections.abc import Iterator

from Interfaces import IHeadMass, CreatingMass

from .Mass import Mass


class HeadMass(IHeadMass, Mass):
    def __init__(self) -> None:
        super().__init__()
        
    def __MakeMassLine(self,width) -> Iterator[Mass]:
        if(width >= 0): yield Mass()
        
    def MakeBoard(self, width: int, height: int) -> list[list[CreatingMass]]:
        
        board: list[list[CreatingMass]] = [
            [ 
             Mass() for i in range(width)
            ]
            for j in range(height)
        ]
        
        board[0][0] = self
        
        print(board)
        
        for i in range(height-1):
            for j in range(width-1):
                board[i][j].Right = board[i][j+1]
                board[i][j].Bottom = board[i+1][j]
                
        for i in range(width-1):
            board[height-1][i].Right = board[height-1][i+1]
                
                
        self.AddAccessor()
        
        return board