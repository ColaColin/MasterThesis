import abc
from core.solved.GameSolver import GameSolver
import subprocess

from utils.prints import logMsg

import time

import os

from impls.solved.pons.pons import PascalPonsSolver

def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

class PonsSolver(GameSolver, metaclass=abc.ABCMeta):
    """
    Uses the connect4 solver made by Pascal Pons.
    Direct calls into a cython interface which skips the expensive reset() call that the command line version does.
    """

    def __init__(self, book, executable = None, mode = "strong"):
        self.executable = executable
        self.book = book
        self.process = None
        self.calls = 0
        self.mode = mode
        self.csolver = None
 
    def restart(self):
        if not (self.process is None):
            self.process.kill()
        self.process = subprocess.Popen([self.executable, "-b", self.book], 
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1)
        self.process.stdout.readline()

    def weakenSolution1(self, solution):
        result = dict()
        for m in solution:
            s = solution[m]
            if s > 0:
                result[m] = 1
            else:
                result[m] = s 
        return result

    def weakenSolution2(self, solution):
        result = dict()
        for m in solution:
            s = solution[m]
            if s > 0:
                result[m] = 1
            elif s < 0:
                result[m] = -1
            else:
                result[m] = 0
        return result

    def getDirectCallResult(self, path):
        if self.csolver is None:
            self.csolver = PascalPonsSolver()
            self.csolver.loadBook(self.book)
        return -self.csolver.solve(path)

    def getPonsResultSubprocess(self, path):
        if self.process is None:
            self.restart()

        self.process.stdin.write(path + "\n")
        ponsOut = self.process.stdout.readline().strip()
        if ponsOut.find("Line") != 0:
            score = ponsOut.split(" ")
            if len(score) == 4 and isInt(score[1]):
                return -int(score[1])
        
        raise Exception("invalid input")

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
                path = "".join([str(p + 1) for p in path])

                #subprocResult = self.getPonsResultSubprocess(path)

                directCallResult = self.getDirectCallResult(path)

                result[move] = directCallResult

        #print("".join([str(m + 1) for m in movesReplay]), result)

        if self.mode == "weak1":
            return self.weakenSolution1(result)
        elif self.mode == "weak2":
            return self.weakenSolution2(result)
        else:
            return result
        