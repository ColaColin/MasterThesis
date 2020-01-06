import unittest
import abc
from core.testcase.abstract.policyiterator import TestPolicyIterationSanity

from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator

class MCTS0Test(unittest.TestCase, TestPolicyIterationSanity, metaclass=abc.ABCMeta):
    def setUp(self):
        self.subject = MctsPolicyIterator(1800, 1.5, 0.1, 0.1)

