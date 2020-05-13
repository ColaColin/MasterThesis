import setproctitle
import time
from utils.prints import logMsg, setLoggingEnabled

from utils.req import requestJson, postJson

import tempfile

from core.mains.mlconfsetup import loadMlConfig

from core.training.NetworkApi import NetworkApi

from impls.selfplay.movedeciders import TemperatureMoveDecider
from core.solved.PolicyTester import PolicyPlayer, PolicyIteratorPlayer, DatasetPolicyTester2

import sys

# python -m core.mains.remote_evaluator --command https://x0.cclausen.eu --secret 42

if __name__ == "__main__":
    setproctitle.setproctitle("x0_remote_evaluator")
    setLoggingEnabled(True)

    datasetPath = "datasets/connect4/testset.txt.zip"

    hasArgs = ("--secret" in sys.argv) and ("--command" in sys.argv)

    if not hasArgs:
        raise Exception("You need to provide arguments for the remote evaluator: --secret <server password> and --command <command server host>!")

    secret = sys.argv[sys.argv.index("--secret")+1]
    commandHost = sys.argv[sys.argv.index("--command")+1]

    networks = NetworkApi(noRun=True)

    def getNextWork():
        while True:
            nextWork = requestJson(commandHost + "/api/evaluations/", secret)
            if len(nextWork) > 0:
                return nextWork[0]["run"], nextWork[0]["network"]
            time.sleep(15)

    def getRunConfig(runId):
        """
        returns a path to a temporary file, which contains the run config.
        Use "with getRunConfig as configFile":
        """

        runConfig = ""
        while True:
            try:
                runConfig = requestJson(commandHost + "/api/runs/" + runId, secret)["config"]
                break
            except Exception as error:
                logMsg("Could not get run configuration for run, will try again soon", error)
                time.sleep(15)

        ff = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+")
        ff.write(runConfig)
        ff.flush()

        logMsg("Using tempfile for configuration:", ff.name)

        return ff

    def submitResult(result, network_id):
        postJson(commandHost + "/api/evaluations/" + network_id, secret, result)

    while True:
        run, network = getNextWork()

        logMsg("Next work: run=%s, network=%s" % (run, network))

        with getRunConfig(run) as temp:
            core = loadMlConfig(temp.name)
        
        policy = core.worker.policy(recursive=True)
        unpackedNetwork = networks.downloadNetwork(network)
        policy.load(unpackedNetwork)

        logMsg("Evaluation: Loaded policy with id", policy.getUUID())

        policyIterator = core.worker.policyIterator(recursive=True)

        # pick the best moves moveDecider
        moveDecider = TemperatureMoveDecider(-1)

        initialState = core.worker.initialState(recursive=True)

        networkPlayer = PolicyPlayer(policy, None, moveDecider)
        fullPlayer = PolicyIteratorPlayer(policy, policyIterator, None, moveDecider, 128)

        networkMoveAcc, networkWinAcc = DatasetPolicyTester2(networkPlayer, datasetPath, initialState).main()
        mctsMoveAcc, _ = DatasetPolicyTester2(fullPlayer, datasetPath, initialState).main()

        submitResult({
            "acc_network_moves": networkMoveAcc,
            "acc_network_wins": networkWinAcc,
            "acc_mcts_moves": mctsMoveAcc
        }, network)


