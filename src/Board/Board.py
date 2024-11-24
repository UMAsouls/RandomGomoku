from injector import inject

from Interfaces import IHeadMass

class Board():
    
    @inject
    def __init__(self, headmass: IHeadMass) -> None:
        self.headmass: IHeadMass = headmass