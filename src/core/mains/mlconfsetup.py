import mlconfig

from impls.selfplay.LinearSelfPlay import LinearSelfPlayWorker
from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator
from impls.selfplay.movedeciders import TemperatureMoveDecider
from impls.games.mnk.mnk import MNKGameState
from impls.games.connect4.connect4 import Connect4GameState
from impls.singleprocess.singleprocess import SingleProcessReporter, SingleProcessUpdater, FilePolicyUpdater, NoopPolicyUpdater, NoopGameReporter
from impls.polices.pytorch.policy import PytorchPolicy, LrStepSchedule, OneCycleSchedule
from core.playing.playvs import PlayVs
from impls.externalplayers.human import HumanMNKInterface, HumanConnect4Interface
from impls.solved.players import RandomPlayPolicy, BestPlayPolicy, SemiPerfectPolicy
from impls.solved.PonsSolver import PonsSolver
from core.solved.TestDatabaseGenerator import TestDatabaseGenerator, TestDatabaseGenerator2
from core.solved.PolicyTester import PolicyPlayer, ShuffleBatchedPolicyPlayer, SolverBatchedPolicyPlayer, PolicyIteratorPlayer, DatasetPolicyTester, DatasetPolicyTester2
from impls.distributed.distributed import DistributedNetworkUpdater, DistributedReporter, DistributedNetworkUpdater2
from core.training.TrainingWorker import TrainingWorker
from impls.training.ConstantTrainingWindowManager import ConstantTrainingWindowManager
from core.training.StreamTrainingWorker import StreamTrainingWorker
from core.training.StreamTrainingWorker import ConstantWindowSizeManager, SlowWindowSizeManager
from core.solved.supervised import SupervisedNetworkTrainer
from core.training.StreamTrainingWorker2 import StreamTrainingWorker2
from impls.selfplay.PlayersSelfplay import LeagueSelfPlayerWorker, FixedPlayerAccess, FixedThinkDecider, LeaguePlayerAccess, LearntThinkDecider
from impls.selfplay.TreeSelfPlay import TreeSelfPlayWorker, FakeEvaluationAccess
from core.command.leagues import EloGaussServerLeague

import sys
from utils.prints import logMsg, setLoggingEnabled

registered = False

def registerClasses():
    global registered
    if not registered:
        mlconfig.register(FakeEvaluationAccess)
        mlconfig.register(TreeSelfPlayWorker)
        mlconfig.register(LearntThinkDecider)
        mlconfig.register(EloGaussServerLeague)
        mlconfig.register(LeaguePlayerAccess)
        mlconfig.register(FixedPlayerAccess)
        mlconfig.register(FixedThinkDecider)
        mlconfig.register(LeagueSelfPlayerWorker)
        mlconfig.register(NoopPolicyUpdater)
        mlconfig.register(NoopGameReporter)
        mlconfig.register(DatasetPolicyTester)
        mlconfig.register(ShuffleBatchedPolicyPlayer)
        mlconfig.register(SolverBatchedPolicyPlayer)
        mlconfig.register(PolicyPlayer)
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
        mlconfig.register(SemiPerfectPolicy)
        mlconfig.register(TestDatabaseGenerator2)
        mlconfig.register(DatasetPolicyTester2)
        mlconfig.register(FilePolicyUpdater)
        mlconfig.register(SupervisedNetworkTrainer)
        mlconfig.register(DistributedNetworkUpdater2)
        mlconfig.register(StreamTrainingWorker2)
        mlconfig.register(OneCycleSchedule)
        mlconfig.register(SlowWindowSizeManager)
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