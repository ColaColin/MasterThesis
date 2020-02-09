import unittest
import abc
from core.testcases.abstract.policyiterator import TestPolicyIterationSanity

from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator

class MCTS0Test(unittest.TestCase, TestPolicyIterationSanity, metaclass=abc.ABCMeta):
    def setUp(self):
        # low cpuct so winner prob tests are a bit easier to pass
        # fpu of 0.5 is better to use perfect winner prediction while not running
        # after a completely wrong move prediction in the usesWinnerOraclePolicy test
        # that also shows how the fpu of 0 likely is a suboptimal choice and it should probably somehow
        # be a less constant value
        self.subject = MctsPolicyIterator(1800, 0.95, 0.0, 0.1, 0.5)

