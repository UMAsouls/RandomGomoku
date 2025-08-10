from abc import ABC, abstractmethod
import numpy as np

class IEnv(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def step(self, action: tuple[int]) -> None:
        pass
    
    @abstractmethod
    def GetBoard(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def OpposeBoard(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def SetBoard(self, board: np.ndarray) -> None:
        pass
    
    