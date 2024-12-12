from random import randint



class RandomSetter:
    def __init__(self, x:int, y:int, w:int, h:int) -> None:
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h
        
        
    def RandomMassGet(self) -> tuple[int,int]:
        sx = self.x + randint(0,self.w-1)
        sy = self.y + randint(0,self.h-1)
        
        return (sx,sy)