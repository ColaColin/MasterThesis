import unittest
import abc
from core.testcase.abstract.gamestate import TestGameStateSanity

from impls.games.connect4.connect4 import Connect4GameState

class Connect4Test(unittest.TestCase, TestGameStateSanity, metaclass=abc.ABCMeta):
    
    def setUp(self):
        self.subject = Connect4GameState(7, 6, 4)

    def getExampleGameSequences(self):
        examples = []

        examples.append(([4, 4, 4, 5, 3, 3, 2, 5, 1], 9, 1))
        examples.append(([6, 5, 6, 5, 6, 5, 4, 5], 8, 2))
        examples.append(([3, 4, 2, 3, 2, 2, 1, 1, 1, 1], 10, 2))

        return examples
