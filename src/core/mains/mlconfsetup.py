import mlconfig

from impls.selfplay.LinearSelfPlay import LinearSelfPlayWorker

from impls.mlconf.mlconf import TypeA, TypeB, TypeX


def registerClasses():
    mlconfig.register(LinearSelfPlayWorker)
    mlconfig.register(TypeA)
    mlconfig.register(TypeB)
    mlconfig.register(TypeX)