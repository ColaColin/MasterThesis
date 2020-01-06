import unittest

from impls.selfplay.movedeciders import TemperatureMoveDecider

import numpy as np

class FakeGame():
    """
    This isn't really a black box test, as we use the knowledge
    that the TemperatureMoveDecider only calls these two methods
    """
    def __init__(self, turn, legalMoves):
        self.legalMoves = legalMoves
        self.turn = turn
    
    def getLegalMoves(self):
        return self.legalMoves.copy()

    def getTurn(self):
        return self.turn

class TestTemperatureMoveDecider(unittest.TestCase):
    def setUp(self):
        self.subject = TemperatureMoveDecider(10)

    def test_pickDeterministic1(self):
        game = FakeGame(11, [0,1,2])
        policy = np.array([0.1, 0.5, 0.3, 0.1])
        move = self.subject.decideMove(game, policy, None)
        self.assertEqual(move, 1)
    
    def test_pickDeterministic2(self):
        game = FakeGame(22, [42, 43])
        policy = np.array(([0] * 42) + [0.1, 0.8, 0, 0, 0.1])
        move = self.subject.decideMove(game, policy, None)
        self.assertEqual(move, 43)
    
    def test_pickProbabilistic1(self):
        game = FakeGame(3, [0,1,2])
        policy = np.array([0.1, 0.5, 0.3, 0.1])
        runs = 1000
        moves = [self.subject.decideMove(game, policy, None) for _ in range(runs)]
        ones = len([x for x in moves if x == 1])
        self.assertTrue(ones > runs * 0.4)

    def test_pickProbabilistic2(self):
        game = FakeGame(3, [42, 43])
        policy = np.array(([0] * 42) + [0.1, 0.8, 0, 0, 0.1])
        runs = 1000
        moves = [self.subject.decideMove(game, policy, None) for _ in range(runs)]
        hits = len([x for x in moves if x == 43])
        self.assertTrue(hits > runs * 0.65)