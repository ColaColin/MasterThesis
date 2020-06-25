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
    ns = []

    for _ in range(BATCH_COUNT):
        tx, ex, n = worker.playBatch()
        times.append(tx)
        exs.append(ex)
        ns.append(n)

    while time.monotonic() - startTime < MIN_TIME:
        tx, ex, n = worker.playBatch()
        times.append(tx)
        exs.append(ex)
        ns.append(n)

    if not None in exs:
        logMsg("Avg number of mcts nodes used by playBatch(): ", np.mean(exs))

    return np.mean(times), np.sum(ns)

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

    def getRunConfig(runId, networkFile, isTreeSelfPlayAr):
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

        ff = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+")

        if "evalAccess" in editConfig["worker"]:
            # TreeSelfPlay, force local evaluation using 4 workers.
            lcKey = "localEvalAccessFrametimeMeasurement"
            editConfig[lcKey] = dict()
            editConfig[lcKey]["name"] = "LocalEvaluationAccess"
            editConfig[lcKey]["workerN"] = PROC_COUNT
            editConfig[lcKey]["forceRun"] = runId
            editConfig[lcKey]["forceCfg"] = ff.name
            editConfig["worker"]["evalAccess"] = "$" + lcKey
            
            if editConfig["worker"]["maxPendingPackages"] < PROC_COUNT * 2:
                editConfig["worker"]["maxPendingPackages"] = PROC_COUNT * 2

            editConfig["evalWorker"]["policyUpdater"] = "$" + plkey
            editConfig["evalWorker"]["isFrameTimeTest"] = True

            isTreeSelfPlayAr.append(True)
        else:
            editConfig["worker"]["policyUpdater"] = "$" + plkey

        yaml.dump(editConfig, ff)
        ff.flush()

        logMsg("Using tempfile for configuration:", ff.name)

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
        run, network = getNextWork()

        pool = mp.Pool(processes=PROC_COUNT)

        logMsg("Next work: run=%s, network=%s" % (run, network))

        time.sleep(3)

        proc = tryPlayersProxyProcess(commandHost, secret, network)

        logMsg("players proxy with specific network should be running now!")

        callResults = []
        genCounts = []

        with getNetworkFile(network) as networkFile:
            isTreeSelfPlayAr = []
            with getRunConfig(run, networkFile.name, isTreeSelfPlayAr) as config:
                isTreeSelfPlay = len(isTreeSelfPlayAr) > 0

                startTime = time.monotonic()

                if isTreeSelfPlay:
                    logMsg("Frametime measurement for tree self play!")
                    callResults.append(pool.apply_async(measureFrametime, (config.name, 0, run)))
                else:
                    logMsg("Frametime measurement for normal play!")
                    for idx in range(PROC_COUNT):
                        callResults.append(pool.apply_async(measureFrametime, (config.name, idx, run)))
                
                plainResults = list(map(lambda x: x.get(), callResults))

                finishTime = time.monotonic()

                genCounts = list(map(lambda x: x[1], plainResults))
                callResults = list(map(lambda x: 1000 / x[0], plainResults))

        frametime = 1000 / sum(callResults)

        workTime = (finishTime - startTime) * 1000
        ftByGenCount =  workTime / np.sum(genCounts)

        logMsg("Work completed, measured frametime of %s for network %s. Generated %i examples in %.2f seconds. frametime by generation count is thus %.2f!" % (frametime, network, np.sum(genCounts), (finishTime - startTime), ftByGenCount))

        submitResult(ftByGenCount, network)

        time.sleep(5)

        pool.terminate()
        proc.wait()

        exit(0)





