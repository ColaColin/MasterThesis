import unittest
import abc
from core.testcases.abstract.policy import TestPolicySanity

from impls.polices.pytorch.policy import PytorchPolicy, LrStepSchedule
from impls.games.mnk.mnk import MNKGameState
from utils.prints import setLoggingEnabled

import torch
import torch.cuda

import sys

class PytorchPolicyTest(unittest.TestCase, TestPolicySanity, metaclass=abc.ABCMeta):
    def setUp(self):
        lrDecider = LrStepSchedule(0.025, 50, 0.1, 0.0001)
        self.subject = PytorchPolicy(7, 1, 16, 1, 1, self.getExampleGameState(),\
            "cuda" if torch.cuda.is_available else "cpu", "torch.optim.adamw.AdamW",\
                None, extraHeadFilters = 32, lrDecider=lrDecider, networkMode="sq",
                replyWeight=0.35)
        
    def getExampleGameState(self):
        return MNKGameState(3, 3, 3)

    def test_mergeInto(self):
        prntVerbose = ('-v' in sys.argv) or ('--verbose' in sys.argv)
        setLoggingEnabled(prntVerbose)

        exampleA, exampleB = self.makeEqualExamples()
        batcher = self.subject.getExamplePrepareObject()

        aMoves = exampleA[1]
        bMoves = exampleB[1]

        aWins = exampleA[2]
        bWins = exampleB[2]

        aMovesBefore = torch.clone(aMoves)
        aWinsBefore = torch.clone(aWins)

        batcher.mergeInto(exampleA, exampleB, 0.6)

        self.assertTrue(torch.equal(exampleA[1], aMovesBefore * 0.4 + 0.6 * exampleB[1]))
        self.assertTrue(torch.equal(exampleA[2], aWinsBefore * 0.4 + 0.6 * exampleB[2]))