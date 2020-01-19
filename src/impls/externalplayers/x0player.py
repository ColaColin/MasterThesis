import abc

from core.playing.playvs import ExternalPlayerInterface

class X0Player(ExternalPlayerInterface, metaclass=abc.ABCMeta):
    
    def showGame(self, gameState):
        pass

    def getMove(self, gameState):
        # TODO implement this
        pass

    def showFinalResult(self, externalId, gameState):
        pass
