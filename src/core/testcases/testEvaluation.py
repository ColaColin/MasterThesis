import unittest
import abc

from impls.games.connect4.connect4 import Connect4GameState
from impls.solved.PonsSolver import PonsSolver
import tempfile
from core.solved.TestDatabaseGenerator import TestDatabaseGenerator2 
from impls.solved.players import SemiPerfectPolicy
from core.solved.PolicyTester import DatasetPolicyTester2, SolverBatchedPolicyPlayer, BatchedPolicyPlayer
from core.solved.TestDatabaseGenerator import getBestScoreKeys
import random

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


class ControlledPlayer(BatchedPolicyPlayer, metaclass=abc.ABCMeta):
    def __init__(self, solver, k):
        self.solver = solver
        self.n = 1
        self.k = k
    
    def getMoves(self, batch):
        result = []
        for state, history in batch:
            assert not state.hasEnded()
            scores = self.solver.getMoveScores(state, history)
            keys = getBestScoreKeys(scores)

            expectedResult = sign(scores[keys[0]])

            if self.n % self.k == 0:
                if len(keys) > 0:
                    result.append((random.choice(keys), expectedResult))
                else:
                    result.append((state.getLegalMoves()[0], 0))
            else:
                legals = state.getLegalMoves()
                badMove = legals[0]
                for m in legals:
                    if not (m in keys):
                        badMove = m
                        break
                result.append((badMove, expectedResult + 1))

            self.n += 1
        return result

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.tempFile = tempfile.NamedTemporaryFile(suffix=".txt.zip", mode="w+")

    def tearDown(self):
        self.tempFile.close()


    def test_generateDatasetAndSolvePerfectly(self):
        game = Connect4GameState()
        solver = PonsSolver("../pons/7x6.book", mode="strong")
        generator = TestDatabaseGenerator2(game, solver, 300, self.tempFile.name, SemiPerfectPolicy(0.5), True, 4, True)
        generator.main(timeout=8)
        self.assertEqual(len(generator.results), 300)

        tester = DatasetPolicyTester2(SolverBatchedPolicyPlayer(solver), self.tempFile.name, game)
        moveAccuracy, resultAccuracy = tester.main()

        self.assertEqual(moveAccuracy, 100)
        self.assertEqual(resultAccuracy, 100)

    def test_halfperfect(self):
        game = Connect4GameState()
        solver = PonsSolver("../pons/7x6.book", mode="strong")
        generator = TestDatabaseGenerator2(game, solver, 300, self.tempFile.name, SemiPerfectPolicy(0.5), True, 4, True)
        generator.main(timeout=8)
        self.assertEqual(len(generator.results), 300)

        tester = DatasetPolicyTester2(ControlledPlayer(solver, 2), self.tempFile.name, game)
        moveAccuracy, resultAccuracy = tester.main()

        self.assertEqual(moveAccuracy, 50)
        self.assertEqual(resultAccuracy, 50)

    def test_nothingCorrect(self):
        game = Connect4GameState()
        solver = PonsSolver("../pons/7x6.book", mode="strong")
        generator = TestDatabaseGenerator2(game, solver, 300, self.tempFile.name, SemiPerfectPolicy(0.5), True, 4, True)
        generator.main(timeout=8)
        self.assertEqual(len(generator.results), 300)

        tester = DatasetPolicyTester2(ControlledPlayer(solver, 99999), self.tempFile.name, game)
        moveAccuracy, resultAccuracy = tester.main()

        self.assertEqual(moveAccuracy, 0)
        self.assertEqual(resultAccuracy, 0)