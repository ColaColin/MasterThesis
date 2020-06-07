import setproctitle
import time
from utils.prints import logMsg, setLoggingEnabled

import sys
from core.training.NetworkApi import NetworkApi

from utils.req import requestJson, postJson

import tempfile

import ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True

from utils.bsonHelp.bsonHelp import decodeFromBson, encodeToBson

import multiprocessing as mp

import numpy as np

from core.mains.mlconfsetup import loadMlConfig

from core.mains.players_proxy import tryPlayersProxyProcess

# python -m core.mains.frametime_evaluator --command https://x0.cclausen.eu --secret 42

#make sure the gpu is 100% loaded by using enough worker threads!
PROC_COUNT = 4
BATCH_COUNT = 60
MIN_TIME = 180

def measureFrametime(configPath, idx, run):
    setproctitle.setproctitle("x0_fe_worker_" + str(idx))
    startTime = time.monotonic()

    core = loadMlConfig(configPath)
    setLoggingEnabled(True)

    worker = core.worker(recursive=True)

    worker.initSelfplay(run)

    times = []
    exs = []

    for _ in range(BATCH_COUNT):
        tx, ex = worker.playBatch()
        times.append(tx)
        exs.append(ex)

    while time.monotonic() - startTime < MIN_TIME:
        tx, ex = worker.playBatch()
        times.append(tx)
        exs.append(ex)

    if not None in exs:
        logMsg("Avg number of mcts nodes used by playBatch(): ", np.mean(exs))

    return np.mean(times)

if __name__ == "__main__":
    setproctitle.setproctitle("x0_frametime_evaluator")
    setLoggingEnabled(True)

    hasArgs = ("--secret" in sys.argv) and ("--command" in sys.argv)

    if not hasArgs:
        raise Exception("You need to provide arguments for the remote evaluator: --secret <server password> and --command <command server host>!")

    secret = sys.argv[sys.argv.index("--secret")+1]
    commandHost = sys.argv[sys.argv.index("--command")+1]

    networks = NetworkApi(noRun=True)

    def getNextWork():
        while True:
            nextWork = requestJson(commandHost + "/api/frametimes/", secret)
            if len(nextWork) > 0:
                return nextWork[0]["run"], nextWork[0]["network"]
            time.sleep(15)

    def getRunConfig(runId, networkFile):
        """
        returns a path to a temporary file, which contains the run config, modified to load the network from the given path,
        and to use a noop game reporter.
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

        grkey = "noopGameReporter23233"
        plkey = "filePolicyLoader1212125"

        editConfig = yaml.load(runConfig)
        editConfig[grkey] = dict()
        editConfig[grkey]["name"] = "NoopGameReporter"

        editConfig[plkey] = dict()
        editConfig[plkey]["name"] = "FilePolicyUpdater"
        editConfig[plkey]["path"] = networkFile

        editConfig["worker"]["gameReporter"] = "$" + grkey
        editConfig["worker"]["policyUpdater"] = "$" + plkey

        ff = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+")
        yaml.dump(editConfig, ff)
        ff.flush()

        #logMsg("Using tempfile for configuration:", ff.name)

        return ff

    def getNetworkFile(networkId):
        networkData = encodeToBson(networks.downloadNetwork(network))

        ff = tempfile.NamedTemporaryFile(suffix=".x0", mode="wb+")
        ff.write(networkData)
        ff.flush()

        #logMsg("Using tempfile for network:", ff.name)

        return ff

    def submitResult(frametime, network_id):
        result = {
            "frametime": frametime
        }
        postJson(commandHost + "/api/frametimes/" + network_id, secret, result)
    
    while True:
        pool = mp.Pool(processes=PROC_COUNT)

        run, network = getNextWork()

        logMsg("Next work: run=%s, network=%s" % (run, network))

        time.sleep(3)

        proc = tryPlayersProxyProcess(commandHost, secret, network)

        logMsg("players proxy with specific network should be running now!")

        callResults = []

        with getNetworkFile(network) as networkFile:
            with getRunConfig(run, networkFile.name) as config:
                for idx in range(PROC_COUNT):
                    callResults.append(pool.apply_async(measureFrametime, (config.name, idx, run)))
                callResults = list(map(lambda x: 1000 / x.get(), callResults))

        frametime = 1000 / sum(callResults)

        logMsg("Work completed, measured frametime of %s for network %s" % (frametime, network))

        submitResult(frametime, network)

        time.sleep(5)

        pool.terminate()
        proc.wait()





