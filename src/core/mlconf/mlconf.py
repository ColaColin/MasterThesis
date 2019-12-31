import abc

#this was used to do some tests on how to use the mlconf library in combination with cython and abc

class ITypeA(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def doStuff(self):
        """
        bla bla bla
        """

class ITypeB(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def wtf(self):
        """
        bla bla bla
        """

class ITypeX(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def doStuff(self):
        """
        bla bla bla
        """