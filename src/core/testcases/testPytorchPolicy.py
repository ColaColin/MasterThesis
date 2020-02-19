import unittest
import abc
from core.testcases.abstract.policy import TestPolicySanity

from impls.polices.pytorch.policy import PytorchPolicy
from impls.games.mnk.mnk import MNKGameState

import torch.cuda

class PytorchPolicyTest(unittest.TestCase, TestPolicySanity, metaclass=abc.ABCMeta):
    def setUp(self):
        self.subject = PytorchPolicy(7, 1, 16, 1, 1, self.getExampleGameState(),\
            "cuda:0" if torch.cuda.is_available else "cpu", "torch.optim.adamw.AdamW", dict(), extraHeadFilters = 32)
        
    def getExampleGameState(self):
        return MNKGameState(3, 3, 3)