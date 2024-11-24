from abc import ABC, abstractmethod

class IMass(ABC):
    @abstractmethod
    def SetStone(self) -> None:
        pass