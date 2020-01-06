import mlconfig

from impls.selfplay.LinearSelfPlay import LinearSelfPlayWorker
from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator
from impls.selfplay.movedeciders import TemperatureMoveDecider
from impls.games.mnk.mnk import MNKGame
from impls.mlconf.mlconf import TypeA, TypeB, TypeX


def registerClasses():
    mlconfig.register(TypeA)
    mlconfig.register(TypeB)
    mlconfig.register(TypeX)

    mlconfig.register(LinearSelfPlayWorker)
    mlconfig.register(MctsPolicyIterator)
    mlconfig.register(TemperatureMoveDecider)
    mlconfig.register(MNKGame)