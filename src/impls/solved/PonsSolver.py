import abc
from core.solved.GameSolver import GameSolver
import subprocess

from utils.prints import logMsg

import time

def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

class PonsSolver(GameSolver, metaclass=abc.ABCMeta):
    """
    Uses the connect4 solver made by Pascal Pons.
    """

    def __init__(self, executable, book):
        """
        using threads above 1 does not help at all, probably because of
        the solver using some sort of position caching?! Or maybe
        the inter process communication is the bottleneck
        """
        self.executable = executable
        self.book = book
        self.process = None
        self.calls = 0
        self.restart()
 
    def restart(self):
        if not (self.process is None):
            self.process.kill()
        self.process = subprocess.Popen([self.executable, "-b", self.book], 
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1)
        self.process.stdout.readline()

    def getMoveScores(self, state, movesReplay):
        assert not state.hasEnded()

        lMoves = state.getLegalMoves()
        result = dict()

        for move in lMoves:
            movePlayed = state.playMove(move)
            if movePlayed.hasEnded():
                if movePlayed.getWinnerNumber() == state.getPlayerOnTurnNumber():
                    result[move] = 99
                else:
                    result[move] = 0
            else:
                path = movesReplay + [move]
                path = "".join([str(p + 1) for p in path]) + "\n"

                self.process.stdin.write(path)
                ponsOut = self.process.stdout.readline().strip()

                #print(ponsOut)

                if ponsOut.find("Line") != 0:
                    score = ponsOut.split(" ")
                    if len(score) == 4 and isInt(score[1]):
                        result[move] = -int(score[1])

        #print("".join([str(m + 1) for m in movesReplay]), result)

        return result