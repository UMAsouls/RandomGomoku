from abc import ABC, abstractmethod

from RandomGomoku.const import Stone

class IMass(ABC):
    @abstractmethod
    def SetStone(self, stone: Stone) -> bool:
        pass
    
    @abstractmethod
    def GetStatus(self) -> int:
        pass
    
    @property
    @abstractmethod
    def Topleft(self) -> "IMass":
        pass
    
    @property
    @abstractmethod
    def Top(self) -> "IMass":
        pass
    
    @property
    @abstractmethod
    def Topright(self) -> "IMass":
        pass
    
    @property
    @abstractmethod
    def Left(self) -> "IMass":
        pass
    
    @property
    @abstractmethod
    def Right(self) -> "IMass":
        pass
    
    @property
    @abstractmethod
    def Bottomleft(self) -> "IMass":
        pass
    
    @property
    @abstractmethod
    def Bottom(self) -> "IMass":
        pass
    
    @property
    @abstractmethod
    def Bottomright(self) -> "IMass":
        pass