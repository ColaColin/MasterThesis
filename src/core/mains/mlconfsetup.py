import mlconfig

from impls.selfplay.LinearSelfPlay import LinearSelfPlayWorker
from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator
from impls.selfplay.movedeciders import TemperatureMoveDecider
from impls.games.mnk.mnk import MNKGameState
from impls.games.connect4.connect4 import Connect4GameState
from impls.singleprocess.singleprocess import SingleProcessReporter, SingleProcessUpdater
from impls.polices.pytorch.policy import PytorchPolicy
from core.playing.playvs import PlayVs
from impls.externalplayers.human import HumanMNKInterface, HumanConnect4Interface
from impls.solved.players import RandomPlayPolicy, BestPlayPolicy
from impls.solved.PonsSolver import PonsSolver
from core.solved.TestDatabaseGenerator import TestDatabaseGenerator
from core.solved.PolicyTester import ShuffleBatchedPolicyPlayer, SolverBatchedPolicyPlayer, PolicyIteratorPlayer, DatasetPolicyTester

import sys
from utils.prints import logMsg, setLoggingEnabled

def registerClasses():
    mlconfig.register(DatasetPolicyTester)
    mlconfig.register(ShuffleBatchedPolicyPlayer)
    mlconfig.register(SolverBatchedPolicyPlayer)
    mlconfig.register(PolicyIteratorPlayer)
    mlconfig.register(LinearSelfPlayWorker)
    mlconfig.register(MctsPolicyIterator)
    mlconfig.register(TemperatureMoveDecider)
    mlconfig.register(MNKGameState)
    mlconfig.register(SingleProcessReporter)
    mlconfig.register(SingleProcessUpdater)
    mlconfig.register(PytorchPolicy)
    mlconfig.register(dict)
    mlconfig.register(PlayVs)
    mlconfig.register(HumanMNKInterface)
    mlconfig.register(Connect4GameState)
    mlconfig.register(HumanConnect4Interface)
    mlconfig.register(RandomPlayPolicy)
    mlconfig.register(PonsSolver)
    mlconfig.register(TestDatabaseGenerator)
    mlconfig.register(BestPlayPolicy)


def mlConfigBasedMain():
    setLoggingEnabled(True)
    registerClasses()
    configPath = sys.argv[1]
    logMsg("Running  ", str(sys.argv))

    config = mlconfig.load(configPath)

    return config