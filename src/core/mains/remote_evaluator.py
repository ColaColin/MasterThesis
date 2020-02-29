import setproctitle
import time
from utils.prints import logMsg, setLoggingEnabled

import requests

import tempfile

from core.mains.mlconfsetup import loadMlConfig

from core.training.NetworkApi import NetworkApi

from impls.selfplay.movedeciders import TemperatureMoveDecider
from core.solved.PolicyTester import PolicyIteratorPlayer, DatasetPolicyTester

import sys

# python -m core.mains.remote_evaluator --command https://x0.cclausen.eu --secret 42

if __name__ == "__main__":
    setproctitle.setproctitle("x0_remote_evaluator")
    setLoggingEnabled(True)

    bestDataset = "datasets/connect4/best_small.dataset"
    rndDataset = "datasets/connect4/rnd_small.dataset"

    hasArgs = ("--secret" in sys.argv) and ("--command" in sys.argv)

    if not hasArgs:
        raise Exception("You need to provide arguments for the remote evaluator: --secret <server password> and --command <command server host>!")

    secret = sys.argv[sys.argv.index("--secret")+1]
    commandHost = sys.argv[sys.argv.index("--command")+1]

    networks = NetworkApi(noRun=True)

    def getNextWork():
        while True:
            try:
                response = requests.get(commandHost + "/api/evaluations/", headers={"secret": secret})
                response.raise_for_status()
                nextWork = response.json()

                if len(nextWork) > 0:
                    return nextWork[0]["run"], nextWork[0]["network"]
            except Exception as error:
                logMsg("Could not check for next evaluations to work on, will try again soon", error)

            time.sleep(15)

    def getRunConfig(runId):
        """
        returns a path to a temporary file, which contains the run config.
        Use "with getRunConfig as configFile":
        """

        runConfig = ""
        while True:
            try:
                response = requests.get(commandHost + "/api/runs/" + runId, headers={"secret": secret})
                response.raise_for_status()
                runConfig = response.json()["config"]
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
        while True:
            try:
                response = requests.post(commandHost + "/api/evaluations/" + network_id, json=result, headers={"secret": secret})
                response.raise_for_status()
                break
            except Exception as error:
                logMsg("Could not submit result, will try again soon", error)
                time.sleep(10)

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

        limitedPlayer = PolicyIteratorPlayer(policy, policyIterator, None, moveDecider, 128, quickFactor=1000000) #rely on the build on limit of the policy to decide what the minimal player looks like
        fullPlayer = PolicyIteratorPlayer(policy, policyIterator, None, moveDecider, 128, quickFactor=1)

        limitedRndResult = DatasetPolicyTester(limitedPlayer, rndDataset, initialState, "shell", 128).main()
        fullRndResult = DatasetPolicyTester(fullPlayer, rndDataset, initialState, "shell", 128).main()

        limitedBestResult = DatasetPolicyTester(limitedPlayer, bestDataset, initialState, "shell", 128).main()
        fullBestResult = DatasetPolicyTester(fullPlayer, bestDataset, initialState, "shell", 128).main()

        submitResult({
            "acc_rnd_limited": limitedRndResult,
            "acc_best_limited": limitedBestResult,
            "acc_rnd_full": fullRndResult,
            "acc_best_full": fullBestResult
        }, network)


