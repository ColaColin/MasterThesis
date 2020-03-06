import abc

from core.solved.TestDatabaseGenerator import TestPlayGeneratorPolicy

import random

class RandomPlayPolicy(TestPlayGeneratorPolicy, metaclass=abc.ABCMeta):
    def decideMoves(self, state, solverMoveScores, bestMoveKeys):
        moves = state.getLegalMoves()
        return moves

class SemiPerfectPolicy(TestPlayGeneratorPolicy, metaclass=abc.ABCMeta):
    def __init__(self, p):
        self.p = p

    def decideMoves(self, state, solverMoveScores, bestMoveKeys):
        if random.random() > self.p:
            moves = state.getLegalMoves()
        else:
            moves = bestMoveKeys.copy()
        return moves

class BestPlayPolicy(TestPlayGeneratorPolicy, metaclass=abc.ABCMeta):
    def decideMoves(self, state, solverMoveScores, bestMoveKeys):
        m = bestMoveKeys.copy()
        return m