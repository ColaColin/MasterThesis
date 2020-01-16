import abc
from core.solved.GameSolver import GameSolver
import subprocess

from utils.prints import logMsg

import time

class PonsSolver(GameSolver, metaclass=abc.ABCMeta):
    """
    Uses the connect4 solver made by Pascal Pons.
    """

    def __init__(self, executable, book):
        """
        using threads above 1 does not help at all, probably because of
        the solver using some sort of position caching?!
        """
        self.executable = executable
        self.book = book
        threads = 1
        self.processes = [None] * threads
        self.threads = threads
        self.calls = 0
        self.restart()

    def restartIndex(self, pidx):
        process = self.processes[pidx]
        if process is not None:
            process.kill()
        self.processes[pidx] = subprocess.Popen([self.executable, "-b", self.book], 
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        startupline = self.processes[pidx].stderr.readline()

    def restart(self):
        for pidx in range(len(self.processes)):
            self.restartIndex(pidx)
            # just ignore the startup line...

    def getMoveScores(self, state, movesReplay):
        result = dict()

        self.calls += 1
        if self.calls > 3000:
            self.calls = 0
            self.restart()

        moves = []
        for move in state.getLegalMoves():
            path = movesReplay + [move]
            path = "".join([str(p + 1) for p in path]) + "\n"
            path = bytes(path, encoding="utf8")
            moves.append((move, path))

        while len(moves) > 0:
            pendings = []
            for idx in range(self.threads):
                if len(moves) > 0:
                    next = moves.pop(0)
                    _, path = next
                    doWork = True
                    while doWork:
                        try:
                            self.processes[idx].stdin.write(path)
                            self.processes[idx].stdin.flush(timeout=1)
                            doWork = False
                        except:
                            logMsg("Had to fix an issue with solver", idx)
                            # what is going on
                            self.restartIndex(idx)
                            doWork = True
                    pendings.append(next)
            for idx in range(len(pendings)):
                move, path = pendings[idx]
                ponsOut = self.processes[idx].stdout.readline().decode("utf8").strip()
                if ponsOut.find("Line") != 0:
                    score = ponsOut.split(" ")
                    if len(score) == 4:
                        result[move] = int(score[1])

        return result