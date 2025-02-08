from abc import ABC, abstractmethod

class CreatingMass(ABC):
    @abstractmethod
    def AddAccessor(self) -> None:
        pass
    
    @property
    @abstractmethod
    def Right(self, v:"CreatingMass") -> None:
        pass
    
    @property
    @abstractmethod
    def Bottom(self, v:"CreatingMass") -> None:
        pass
    
    
    
    