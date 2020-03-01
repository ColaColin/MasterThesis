import unittest
import abc
from core.testcases.abstract.policy import TestPolicySanity

from impls.polices.pytorch.policy import PytorchPolicy, LrStepSchedule
from impls.games.mnk.mnk import MNKGameState

import torch.cuda

class PytorchPolicyTest(unittest.TestCase, TestPolicySanity, metaclass=abc.ABCMeta):
    def setUp(self):
        lrDecider = LrStepSchedule(0.025, 50, 0.1, 0.0001)
        self.subject = PytorchPolicy(7, 1, 16, 1, 1, self.getExampleGameState(),\
            "cuda" if torch.cuda.is_available else "cpu", "torch.optim.adamw.AdamW", None, extraHeadFilters = 32, lrDecider=lrDecider)
        
    def getExampleGameState(self):
        return MNKGameState(3, 3, 3)