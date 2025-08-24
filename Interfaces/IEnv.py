from abc import ABC, abstractmethod
import numpy as np

from const import EnvBackup

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
    def GetBoard_Player1(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def GetBoard_Player2(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def GetBoard_CurrentPlayer(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def GetLegalAction(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def backup(self) -> EnvBackup:
        pass
    
    @abstractmethod
    def restore(self, data: EnvBackup):
        pass
    
    @abstractmethod
    def GetCurrentPlayer(self) -> int:
        pass
    
    