import sys

import mlconfig

from utils.prints import logMsg, setLoggingEnabled
from core.mains.mlconfsetup import registerClasses
registerClasses()


# Play games via a SelfPlayWorker implementation
# Report them to somewhere else, while pulling a policy from somewhere else.
# Though the most simple implementation actually trains the policy in here as well
# "Production" level of performance is best reached however by using a distributed setting
# where every server runs as many instances as needed to saturate all resources.
if __name__ == "__main__":
    setLoggingEnabled(True)

    configPath = sys.argv[1]
    logMsg("Running  " + sys.arv)

    # Make a SelfPlayWorker via the config system. That already includes the policy and policy iterator implementation
    # Then run selfplay with a gameReporter and policyUpdater implementation

    config = mlconfig.load(configPath)
    
    selfplayer = config.selfplayer(recursive=True)
    reporter = config.reporter(recursive=True)
    updater = config.updater(recursive=True)

    selfplayer.selfplay(reporter, updater)

    