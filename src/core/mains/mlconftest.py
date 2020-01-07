import mlconfig

from impls.mlconf.mlconf import TypeA, TypeB, TypeX

import mlconfig

from utils.prints import logMsg, setLoggingEnabled

mlconfig.register(TypeA)
mlconfig.register(TypeB)
mlconfig.register(TypeX)


if __name__ == "__main__":
    setLoggingEnabled(False)

    config = mlconfig.load("test.yaml")

    b = config.instanceB(recursive=True)
    b.wtf()
