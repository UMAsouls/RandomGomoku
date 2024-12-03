from abc import ABC, abstractmethod
from .IMass import IMass

class IHeadMass(ABC):
    """Massの左上のところ。Board生成とかを担当する
        
    Args:
        ABC (_type_): _description_
    """
    def MakeBoard(self, width:int, height:int) -> list[list[IMass]]:
        pass