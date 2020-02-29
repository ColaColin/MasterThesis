# meant to run as a process on the command server,
# picking the newest network available in the database, evaluate it and store the evaluation in the database
# all the command server needs to do is start this with the same config file it was started itself with

from core.mains.mlconfsetup import mlConfigBasedMain

import psycopg2
from psycopg2 import pool

import setproctitle
import time

import tempfile

from core.mains.command import getCommandConfiguration

from utils.prints import logMsg, setLoggingEnabled
from utils.bsonHelp.bsonHelp import decodeFromBson

from impls.selfplay.movedeciders import TemperatureMoveDecider
from core.solved.PolicyTester import PolicyIteratorPlayer, DatasetPolicyTester

import os
from core.command.state import getUUIDPath
from utils.misc import readFileUnderPath

if __name__ == "__main__":
    raise Exception("This is old, do not use this anymore, use remote_evaluator instead!")

    setproctitle.setproctitle("x0_evaluator")
    setLoggingEnabled(True)

    logMsg("Started local server evaluator!")

    config = getCommandConfiguration()

    pool = psycopg2.pool.SimpleConnectionPool(1, 3,user = config["dbuser"],
                                            password = config["dbpassword"],
                                            host = "127.0.0.1",
                                            port = "5432",
                                            database = config["dbname"]);

    def getNewestNetwork():
        """
        return (run-id, network-id) of the next network to work on, if none, sleep until there is one
        """
        while True:
            try:
                con = pool.getconn()
                cursor = con.cursor()

                cursor.execute("select run, id from networks where acc_rnd is null order by creation desc")
                rows = cursor.fetchall()

                if len(rows) == 0:
                    logMsg("Waiting for new networks to evaluate!")
                    time.sleep(10)
                else:
                    todo = rows[0]
                    logMsg("Found new network to evaluate!", todo)
                    return todo[0], todo[1]
            finally:
                if cursor:
                    cursor.close()
                pool.putconn(con)

    def getRunConfig(runId):
        """
        returns a path to a temporary file, which contains the run config.
        Use "with getRunConfig as configFile":
        """
        ff = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+")
        try:
            con = pool.getconn()
            cursor = con.cursor()

            cursor.execute("select config from runs where id = %s", (runId, ))
            rows = cursor.fetchall()

            cfg = rows[0][0]
            
            ff.write(cfg)
            ff.flush()

            logMsg("Using tempfile for configuration:", ff.name)

            return ff
        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)

    while True:
        run, network = getNewestNetwork()

        with getRunConfig(run) as temp:
            cfgPath = temp.name
            core = mlConfigBasedMain(cfgPath)

        policy = core.worker.policy(recursive=True)
        networkPath = os.path.join(config["dataPath"], getUUIDPath(network))
        networkData = readFileUnderPath(networkPath)
        unpackedNetwork = decodeFromBson(networkData)
        policy.load(unpackedNetwork)

        logMsg("Evaluation: Loaded policy with id", policy.getUUID())

        policyIterator = core.worker.policyIterator(recursive=True)

        # pick the best moves moveDecider
        moveDecider = TemperatureMoveDecider(-1)

        initialState = core.worker.initialState(recursive=True)

        policyPlayer = PolicyIteratorPlayer(policy, policyIterator, None, moveDecider, 128, quickFactor=config["evaluatorQuickFactor"])

        rndTester = DatasetPolicyTester(policyPlayer, config["testRndMovesDataset"], initialState, "shell", 128)
        rndResult = rndTester.main()

        bestTester = DatasetPolicyTester(policyPlayer, config["testBestMovesDataset"], initialState, "shell", 128)
        bestResult = bestTester.main()

        logMsg("Policy %s tested: %.2f%% on best play, %.2f%% on random play." % (policy.getUUID(), bestResult, rndResult))

        try:
            con = pool.getconn()
            cursor = con.cursor()

            cursor.execute("update networks set acc_rnd = %s, acc_best = %s where id = %s", (rndResult, bestResult, network))
            con.commit()

        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)



    