from injector import Injector, Binder

from RandomGomoku.Interfaces import IHeadMass

from RandomGomoku.Mass import HeadMass


class Dependency():
    def __init__(self):
        self._injector = Injector(self.__class__.config)
        
    @classmethod
    def config(cls, binder:Binder):
        binder.bind(IHeadMass, HeadMass)
        pass
    
    def resolve(self, cls: type):
        return self._injector.get(cls)