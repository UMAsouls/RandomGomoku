
from Interfaces import IMass, IHeadMass
from const import Stone


class Mass(IMass):
    
    
    def __init__(self) -> None:
        
        #方向に合わせたマス
        self.topleft :IMass = None
        self.top :IMass = None
        self.topright :IMass = None
        self.left :IMass = None
        self.right :IMass = None
        self.bottomleft :IMass = None
        self.bottom :IMass = None
        self.bottomright :IMass = None
        
        
        self.around :list[list[IMass]] = [
            [self.topleft, self.top, self.topright],
            [self.left, None, self.right],
            [self.bottomleft, self.bottom, self.bottomright]
        ]
        """マスをidxで取り出しやすくしたもの
        """
        
        
        self.stone :Stone = Stone.NONE
        
        
        
        
    
    
        