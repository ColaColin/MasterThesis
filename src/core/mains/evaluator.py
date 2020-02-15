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

import os
from core.command.state import getUUIDPath, readFileUnderPath

if __name__ == "__main__":
    setproctitle.setproctitle("evaluator")
    setLoggingEnabled(True)

    config = getCommandConfiguration()

    pool = psycopg2.pool.SimpleConnectionPool(1, 20,user = config["dbuser"],
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
                    time.sleep(5000)
                else:
                    todo = rows[0]
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

        policy = core.trainer.policy(recursive=True)
        networkPath = os.path.join(config["dataPath"], getUUIDPath(network))
        networkData = readFileUnderPath(networkPath)
        unpackedNetwork = decodeFromBson(networkData)
        policy.load(unpackedNetwork)

        logMsg("Loaded policy with id", policy.getUUID())


        time.sleep(10)
        


    