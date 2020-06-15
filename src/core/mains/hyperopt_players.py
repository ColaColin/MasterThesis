import setproctitle
from core.mains.mlconfsetup import loadMlConfig

from core.training.NetworkApi import NetworkApi

from impls.selfplay.movedeciders import TemperatureMoveDecider
from core.solved.PolicyTester import PolicyPlayer, PolicyIteratorPlayer, DatasetPolicyTester2

from core.mains.remote_evaluator import getRunConfig
from core.mains.hyperopt import CombinedLogger

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events

from utils.prints import logMsg, setLoggingEnabled

import sys

from utils.req import requestJson

def buildScoreFunction(runId, networkId, networks, datasetPath):
    """
    get the network by ID from the server and evaluate it with the given player, return MCTS accuracy as a score.
    Uses a specific run for the configuration of the system.
    """

    with getRunConfig(runId, commandHost, secret) as temp:
        core = loadMlConfig(temp.name)

    policy = core.worker.policy(recursive=True)
    unpackedNetwork = networks.downloadNetwork(networkId)
    policy.load(unpackedNetwork)

    logMsg("Loaded policy id %s" % policy.getUUID())

    policyIterator = core.worker.policyIterator(recursive=True)

    # pick the best moves moveDecider
    moveDecider = TemperatureMoveDecider(-1)

    initialState = core.worker.initialState(recursive=True)

    def evlFunc(**player):
        nonlocal policy
        nonlocal policyIterator
        nonlocal moveDecider
        nonlocal datasetPath
        nonlocal initialState

        if not "alphaBase" in player:
            player["alphaBase"] = 20.38

        logMsg("Evaluating player", player)

        fullPlayer = PolicyIteratorPlayer(policy, policyIterator, None, moveDecider, 128, playerParams=player)
        mctsMoveAcc, _ = DatasetPolicyTester2(fullPlayer, datasetPath, initialState).main()
        return mctsMoveAcc / 100.0

    return evlFunc

if __name__ == "__main__":
    setproctitle.setproctitle("x0_player_bayesian_opt")
    setLoggingEnabled(False)
    datasetPath = "datasets/connect4/testset.txt.zip"

    hasArgs = ("--secret" in sys.argv) and ("--command" in sys.argv) and ("--run" in sys.argv) and ("--network" in sys.argv)

    if not hasArgs:
        raise Exception("You need to provide arguments for the hyperopt_players: --secret <server password> and --command <command server host> --run <run id> --network <network id>!")

    secret = sys.argv[sys.argv.index("--secret")+1]
    commandHost = sys.argv[sys.argv.index("--command")+1]
    runId = sys.argv[sys.argv.index("--run")+1]
    networkId = sys.argv[sys.argv.index("--network")+1]

    networks = NetworkApi(noRun=True)

    scoreFunc = buildScoreFunction(runId, networkId, networks, datasetPath)

    pbounds = {
        "cpuct": (0, 6),
        "drawValue": (0, 1),
        "fpu": (0, 1)
    }

    optimizer = BayesianOptimization(
        f = scoreFunc,
        pbounds = pbounds,
        random_state= 1
    );

    logger = CombinedLogger("player_opt_logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    #get the best players of the given network, if any
    bestPlayer = requestJson(commandHost + "/api/bestplayer/" + networkId, secret)
    if len(bestPlayer) > 0:
        logMsg("Probing best player of the network!", bestPlayer)
        if "alphaBase" in bestPlayer:
            del bestPlayer["alphaBase"]
        optimizer.probe(
            params=bestPlayer
        )

    # the known "good" parameters of the first hyperopt run
    logMsg("Probing known good parameters!")
    optimizer.probe(
        params={"cpuct": 1.545, "drawValue": 0.6913, "fpu": 0.8545}
    )

    optimizer.maximize(
        init_points=8,
        n_iter=90
    )

    print(optimizer.max)