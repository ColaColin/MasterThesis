import unittest
import abc
from core.testcase.test import TestGameStateSanity

from impls.games.mnk.mnk import MNKGameState

class MNKTest(unittest.TestCase, TestGameStateSanity, metaclass=abc.ABCMeta):
    
    def setUp(self):
        self.subject = MNKGameState(4, 3, 3)
        self.subject.getGameName()

    def callLoad(self, encoded):
        return MNKGameState.load(encoded)

    def getExampleGameSequences(self):
        examples = []

        examples.append(([0, 4, 1, 5, 2], 5, 1))
        examples.append(([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], 11, 1))
        examples.append(([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 2, 11], 12, 2))
        examples.append(([0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 11, 10], 12, 0))

        return examples
