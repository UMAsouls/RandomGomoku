from abc import ABC, abstractmethod
import numpy as np
from RandomGomoku.const import Stone

class IBoard(ABC):
    @abstractmethod
    def MakeBoard(self) -> None:
        pass
    
    @abstractmethod
    def SetBoard(self, board:np.ndarray, opppose:np.ndarray) -> None:
        pass
    
    @abstractmethod
    def GetBoardInt(self) -> np.ndarray:
        pass
    @abstractmethod
    def GetBoardOppose(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def GetStatus(self, x:int, y:int) -> int:
        pass
    
    @abstractmethod
    def SetStone(self, x:int, y:int, stone:Stone) -> int:
        pass
    
    @abstractmethod
    def JudgeWin(self) -> int:
        pass
    
    @abstractmethod
    def StringBoard(self) -> str:
        pass
    
    @abstractmethod
    def PrintBoard(self) -> None:
        pass