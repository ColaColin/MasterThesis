import abc

from core.solved.TestDatabaseGenerator import TestPlayGeneratorPolicy

import random

class RandomPlayPolicy(TestPlayGeneratorPolicy, metaclass=abc.ABCMeta):
    def decideMoves(self, state, solverMoveScores):
        splits = 3 - (state.getTurn() // 4)
        if splits < 1:
            splits = 1
        if state.getTurn() < 2:
            splits = 7 - state.getTurn()
        moves = state.getLegalMoves()
        random.shuffle(moves)
        return moves[:splits]