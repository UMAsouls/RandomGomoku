from injector import Injector, Binder


class Dependency():
    def __init__(self):
        self._injector = Injector(self.__class__.config)
        
    @classmethod
    def config(cls, binder:Binder):
        pass
    
    def resolve(self, cls: type):
        return self._injector.get(cls)