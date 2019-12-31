import mlconfig

from impls.mlconf.mlconf import TypeA, TypeB, TypeX

import mlconfig

mlconfig.register(TypeA)
mlconfig.register(TypeB)
mlconfig.register(TypeX)


if __name__ == "__main__":
    config = mlconfig.load("test.yaml")

    b = config.instanceB(recursive=True)
    b.wtf()
