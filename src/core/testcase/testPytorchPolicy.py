import unittest
import abc
from core.testcase.abstract.policy import TestPolicySanity

from impls.polices.pytorch.policy import PytorchPolicy
from impls.games.mnk.mnk import MNKGameState

import torch.cuda

class PytorchPolicyTest(unittest.TestCase, TestPolicySanity, metaclass=abc.ABCMeta):
    def setUp(self):
        self.subject = PytorchPolicy(64, 1, 16, 1, 4, self.getExampleGameState(), "cuda" if torch.cuda.is_available else "cpu")
        
    def getExampleGameState(self):
        return MNKGameState(3, 3, 3)