import mlconfig

from impls.selfplay.LinearSelfPlay import LinearSelfPlayWorker
from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator
from impls.selfplay.movedeciders import TemperatureMoveDecider
from impls.games.mnk.mnk import MNKGameState
from impls.games.connect4.connect4 import Connect4GameState
from impls.singleprocess.singleprocess import SingleProcessReporter, SingleProcessUpdater
from impls.polices.pytorch.policy import PytorchPolicy, LrStepSchedule
from core.playing.playvs import PlayVs
from impls.externalplayers.human import HumanMNKInterface, HumanConnect4Interface
from impls.solved.players import RandomPlayPolicy, BestPlayPolicy
from impls.solved.PonsSolver import PonsSolver
from core.solved.TestDatabaseGenerator import TestDatabaseGenerator
from core.solved.PolicyTester import ShuffleBatchedPolicyPlayer, SolverBatchedPolicyPlayer, PolicyIteratorPlayer, DatasetPolicyTester
from impls.distributed.distributed import DistributedNetworkUpdater, DistributedReporter
from core.training.TrainingWorker import TrainingWorker
from impls.training.ConstantTrainingWindowManager import ConstantTrainingWindowManager
from core.training.StreamTrainingWorker import StreamTrainingWorker
from core.training.StreamTrainingWorker import ConstantWindowSizeManager

import sys
from utils.prints import logMsg, setLoggingEnabled

registered = False

def registerClasses():
    global registered
    if not registered:
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
        mlconfig.register(DistributedNetworkUpdater)
        mlconfig.register(DistributedReporter)
        mlconfig.register(TrainingWorker)
        mlconfig.register(ConstantTrainingWindowManager)
        mlconfig.register(StreamTrainingWorker)
        mlconfig.register(ConstantWindowSizeManager)
        mlconfig.register(LrStepSchedule)
        registered = True

def mlConfigBasedMain(configPath):
    setLoggingEnabled(True)
    registerClasses()
    logMsg("Running", *sys.argv)

    config = mlconfig.load(configPath)

    return config

def loadMlConfig(path):
    registerClasses()
    return mlconfig.load(path)