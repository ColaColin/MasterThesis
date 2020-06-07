import setproctitle
from utils.prints import logMsg, setLoggingEnabled

import psycopg2
from psycopg2 import pool

import time

from core.mains.command import getCommandConfiguration
from core.command.state import getUUIDPath
from utils.misc import readFileUnderPath
from utils.bsonHelp.bsonHelp import decodeFromBson
from utils.misc import constructor_for_class_name

import numpy as np

import os

if __name__ == "__main__":
    setproctitle.setproctitle("x0_stats_generator")
    
    setLoggingEnabled(True)

    logMsg("Started stats generator!")

    config = getCommandConfiguration()

    pool = psycopg2.pool.SimpleConnectionPool(1, 3,user = config["dbuser"],
                                            password = config["dbpassword"],
                                            host = "127.0.0.1",
                                            port = "5432",
                                            database = config["dbname"]);


    def getNextRunIteration():
        """
        return run-id, max-iteration for a run that is missing stats for finished iterations.
        """
        while True:
            try:
                con = pool.getconn()
                cursor = con.cursor()
                cursor.execute("select id, iterations from runs_info r left join (select run, max(iteration) from run_iteration_stats s group by s.run) s on s.run = r.id where (max is null or (iterations - 1) > max) and iterations > 0")
                rows = cursor.fetchall()

                if len(rows) == 0:
                    logMsg("Waiting for new iterations to generate stats")
                    time.sleep(15)
                else:
                    todo = rows[0]
                    logMsg("Found run that is missing stats!", todo)
                    return todo[0], todo[1]
            finally:
                if cursor:
                    cursor.close()
                pool.putconn(con)
            
    def loadStates(run, iteration):
        """
        Load the states of the given run that are associated with the given iteration,
        cut away as much of the state package as possible however.
        @returns: (stateObject, results, final)[]
        """

        result = []

        startTime = time.monotonic()

        try:
            con = pool.getconn()
            cursor = con.cursor()
            cursor.execute("select id from states where run = %s and iteration = %s", (run, iteration))
            rows = cursor.fetchall()
            for row in rows:
                fpath = os.path.join(config["dataPath"], getUUIDPath(row[0]))
                fbytes = readFileUnderPath(fpath)
                statesPackage = decodeFromBson(fbytes)
                for record in statesPackage:
                    protoState = constructor_for_class_name(record["gameCtor"])(**record["gameParams"])
                    gameState = protoState.load(record["state"])
                    final = False
                    if "final" in record and record["final"]:
                        final = True
                    niters = 0
                    if "numIterations" in record:
                        niters = record["numIterations"]
                    result.append((gameState, record["knownResults"], final, niters))
        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)

        procTime = time.monotonic() - startTime
        logMsg("Loaded %i states for run %s iteration %i in %.2fs" % (len(result), run, iteration, procTime))

        return result

    while True:
        run, iteration = getNextRunIteration()

        statesSet = set()

        curi = 0

        while curi < iteration:
            startIterationTime = time.monotonic()
            logMsg("Analyzing iteration %i of run %s" % (curi, run))
            playedStates = 0
            newStates = 0
            firstPlayersWins = 0
            draws = 0
            resultsCnt = 0

            gameLengths = []
            nodeCounts = []

            states = loadStates(run, curi)

            playedStates += len(states)

            beforeSetSize = len(statesSet)
            for state in states:
                nodeCounts.append(state[3])
                statesSet.add(state[0])
                for result in state[1]:
                    resultsCnt += 1
                    if result == 1:
                        firstPlayersWins += 1
                    if result == 0:
                        draws += 1
                if state[2]:
                    gameLengths.append(state[0].getTurn())

            newStates = len(statesSet) - beforeSetSize

            if len(gameLengths) > 0:
                lengthAvg = float(np.mean(gameLengths))
                lengthStd = float(np.std(gameLengths))
            else:
                lengthAvg = 0
                lengthStd = 0

            if len(nodeCounts) > 0:
                avgNodes = float(np.mean(nodeCounts))
            else:
                avgNodes = 0

            iterationProcTime = time.monotonic() - startIterationTime
            logMsg("Finished analyzing, took %.2fs" % iterationProcTime)

            try:
                con = pool.getconn()
                cursor = con.cursor()
                cursor.execute("delete from run_iteration_stats where run = %s and iteration = %s", (run, curi))
                
                if cursor:
                    cursor.close()
                cursor = con.cursor()                

                fpwins = float((firstPlayersWins / resultsCnt) * 100.0)
                dp = float(draws / resultsCnt) * 100.0

                cursor.execute("insert into run_iteration_stats (run, iteration, played_states, new_states, first_player_wins, draws, game_length_avg, game_length_std, avg_nodes) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (run, curi, playedStates, newStates, fpwins, dp, lengthAvg, lengthStd, avgNodes))

                con.commit()

            finally:
                if cursor:
                    cursor.close()
                pool.putconn(con)

            curi += 1

            if curi == iteration:
                logMsg("Done with current stats work, waiting for continuation!")
                nextRun, nextIteration = getNextRunIteration()
                if run == nextRun and iteration < nextIteration:
                    logMsg("Found continuation until iteration", nextIteration)
                    iteration = nextIteration

