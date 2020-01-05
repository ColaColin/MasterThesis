import mlconfig

from impls.selfplay.LinearSelfPlay import LinearSelfPlayWorker
from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator
from 
from impls.mlconf.mlconf import TypeA, TypeB, TypeX


def registerClasses():
    mlconfig.register(TypeA)
    mlconfig.register(TypeB)
    mlconfig.register(TypeX)

    mlconfig.register(LinearSelfPlayWorker)
    mlconfig.register(MctsPolicyIterator)