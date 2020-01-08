import mlconfig

from impls.mlconf.mlconf import TypeA, TypeB, TypeX

import mlconfig

from utils.prints import logMsg, setLoggingEnabled

from impls.selfplay.LinearSelfPlay import LinearSelfPlayWorker
from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator
from impls.selfplay.movedeciders import TemperatureMoveDecider
from impls.games.mnk.mnk import MNKGameState
from impls.mlconf.mlconf import TypeA, TypeB, TypeX
from impls.singleprocess.singleprocess import SingleProcessReporter, SingleProcessUpdater
from impls.polices.pytorch.policy import PytorchPolicy

mlconfig.register(TypeA)
mlconfig.register(TypeB)
mlconfig.register(TypeX)


if __name__ == "__main__":
    setLoggingEnabled(False)

    # turns out typos in the yaml definition cause confusing errors...

    mnk = MNKGameState(3, 3, 3)
    optimizerArgs = dict([("lr", 0.001), ("weight_decay", 0.001)])
    resnet = PytorchPolicy(128, 1, 16, 1, 1, mnk, "cuda:0", "torch.optim.adamw.AdamW", optimizerArgs)
    wtfMap0 = dict([("expansions", 15), ("cpuct", 1.5), ("rootNoise", 0.2), ("drawValue", 0.5)])
    mcts = MctsPolicyIterator(**wtfMap0)
    tempDecider = TemperatureMoveDecider(12)

    wtfMap1 = dict([("initalState", mnk), ("policy", resnet), ("policyIterator", mcts), ("gameCount", 128), ("moveDecider", tempDecider)])

    selfplayer = LinearSelfPlayWorker(**wtfMap1)
    print(selfplayer)

    # config = mlconfig.load("test.yaml")

    # b = config.instanceB(recursive=True)
    # b.wtf()
