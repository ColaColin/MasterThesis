import unittest
import abc
from core.testcase.abstract.gamestate import TestGameStateSanity

from core.testcase.abstract.policyiterator import TestCountGame

class TestGameTest(unittest.TestCase, TestGameStateSanity, metaclass=abc.ABCMeta):
    """
    The test count game is used in the tests of PolicyIterators
    """
    def setUp(self):
        self.subject = TestCountGame(2, 2, 2, True)
        self.subject.winningPath = [1, 1]
    
    def getExampleGameSequences(self):
        examples = []

        examples.append(([1, 1, 1], 3, 1))
        examples.append(([0, 1, 1, 1], 4, 2))
        examples.append(([0, 0, 0, 0, 0, 1, 1, 1], 8, 2))
        examples.append(([1, 1, 0, 0, 1, 1, 1], 7, 1))

        return examples