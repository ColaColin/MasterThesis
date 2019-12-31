from core.mlconf.mlconf import ITypeA, ITypeB, ITypeX

import abc

class TypeA(ITypeA, metaclass=abc.ABCMeta):
    def __init__(self, param1):
        self.param1 = param1
        print("Create TypeA with param1: ", param1)
    
    def doStuff(self):
        print("Called A:", self.param1)

class TypeX(ITypeX, metaclass=abc.ABCMeta):
    def __init__(self, param1, param2, aInstance):
        self.param1 = param1
        self.param2 = param2
        self.aInstance = aInstance
        print("Create TypeX with param1:", param1, "param2:", param2)

    def doStuff(self):
        print("Called X:", self.param1 + self.param2)
        self.aInstance.doStuff()

class TypeB(ITypeB, metaclass=abc.ABCMeta):
    def __init__(self, param1, aInstance):
        print("Create TypeB with param 1: ", param1)
        aInstance.doStuff()

    def wtf(self):
        print("WTF")