import unittest
import abc
from core.testcases.abstract.policyiterator import TestPolicyIterationSanity

from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator

class MCTS0Test(unittest.TestCase, TestPolicyIterationSanity, metaclass=abc.ABCMeta):
    def setUp(self):
        # low cpuct so winner prob tests are a bit easier to pass
        self.subject = MctsPolicyIterator(1800, 0.95, 0.0, 0.1)

