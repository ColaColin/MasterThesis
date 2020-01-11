import abc
from utils.prints import logMsg

from core.playing.playvs import ExternalPlayerInterface

class ConsolePlayer(ExternalPlayerInterface, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def parseMove(self, gameState, moveStr):
        """
        Parse the move here
        """

    def showGame(self, gameState):
        logMsg("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logMsg(str(gameState))

    def getMove(self, gameState):
        mv = -1
        while mv == -1:
            try:
                mv = self.parseMove(gameState, input("Your turn: "))    
            except:
                pass

        return mv

    def showFinalResult(self, externalId, gameState):
        logMsg("Game ended, Winner: %i" % gameState.getWinnerNumber())
        if gameState.getWinnerNumber() == externalId:
            logMsg("You won!")
        elif gameState.getWinnerNumber() == 0:
            logMsg("Draw!")
        else:
            logMsg("You lost!")

class HumanMNKInterface(ConsolePlayer, metaclass=abc.ABCMeta):

    def parseMove(self, gameState, moveStr):
        ms = moveStr.split("-")
        if len(ms) == 2:
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            cIdx = chars.find(ms[1])
            if cIdx != -1:
                ms[1] = cIdx + 1
                x = int(ms[0]) - 1
                y = int(ms[1]) - 1
                result = y * gameState.getM() + x
                if result in gameState.getLegalMoves():
                    return result
                else:
                    logMsg("That move is illegal")
                    return -1

        logMsg("Cannot parse turn (example of a valid turn is 1-a)")
        return -1


class HumanConnect4Interface(ConsolePlayer, metaclass=abc.ABCMeta):

    def parseMove(self, gameState, moveStr):

        result = int(moveStr) - 1
        if result in gameState.getLegalMoves():
            return result
        else:
            logMsg("That move is illegal")
            return -1

        return -1
