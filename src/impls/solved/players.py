import abc

from core.solved.TestDatabaseGenerator import TestPlayGeneratorPolicy

import random

class RandomPlayPolicy(TestPlayGeneratorPolicy, metaclass=abc.ABCMeta):
    def decideMoves(self, state, solverMoveScores, bestMoveKeys):
        if state.getTurn() < 1:
            splits = 7
        else:
            splits = 3
        moves = state.getLegalMoves()
        random.shuffle(moves)
        return moves[:splits]

class BestPlayPolicy(TestPlayGeneratorPolicy, metaclass=abc.ABCMeta):
    def decideMoves(self, state, solverMoveScores, bestMoveKeys):
        m = bestMoveKeys.copy()
        random.shuffle(m)
        return m