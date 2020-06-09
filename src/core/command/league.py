import falcon
import json
import uuid

from core.mains.mlconfsetup import loadMlConfig
from core.command.runs import RunsResource
import tempfile

import time
from utils.prints import logMsg


def asTempFile(strContent):
    ff = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+")
    ff.write(strContent)
    ff.flush()
    return ff

class LeagueResource():

    def __init__(self, pool):
        self.pool = pool
        self.cachedLeagues = dict()
    
    def loadLeague(self, runId):
        if runId in self.cachedLeagues:
            return self.cachedLeagues[runId]
        else:
            rr = RunsResource(self.pool)
            runConfigString = rr.loadRuns(runId)[0]["config"]
            with asTempFile(runConfigString) as tf:
                core = loadMlConfig(tf.name)

            if hasattr(core, "serverLeague"):
                self.cachedLeagues[runId] = core.serverLeague()
                return self.cachedLeagues[runId]
            else:
                return None

    def on_get(self, req, resp, mode, run_id):
        startGet = time.monotonic()

        league = self.loadLeague(run_id)

        if league is None:
            resp.media = []
        elif mode == "players":
            resp.media = league.getPlayers(self.pool, run_id)
        else:
            allMatches = league.getMatchHistory(self.pool, run_id)
            cutMatches = list(reversed(allMatches))[:30]
            resp.media = cutMatches

        finished = time.monotonic()

        logMsg("league_on_get", mode, run_id, "took", finished - startGet)

        resp.status = falcon.HTTP_200

    def on_post(self, req, resp, mode = None, run_id = None):
        startPost = time.monotonic()

        assert run_id is not None
        league = self.loadLeague(run_id)
        reports = req.media

        league.reportResultBatch(list(map(lambda x: (x["p1"], x["p2"], x["winner"], x["policy"], run_id), reports)), self.pool, run_id)

        finished = time.monotonic()

        resp.status = falcon.HTTP_200

        logMsg("league_on_post", len(reports), mode, run_id, "took", finished - startPost)


def selectBestPlayersForNetwork(pool, netId):
    try:
        con = pool.getconn()

        # the players relevant to the network are the players of the last generation that does not reference a network which was created after the given network
        cursor = con.cursor()
        cursor.execute("select run, creation from networks where id = %s", (netId, ))
        firstRow = cursor.fetchall()[0]
        runId = firstRow[0]
        networkCreationTime = firstRow[1]
        cursor.close()

#run is 1afd62c2-2a0f-4dde-85a6-baa0cbe64be6 network created at 2020-06-08 22:22:34.330666+02:00 for network 758316f8-930a-47e6-b115-e5370a27c7ae
        print("run is %s network created at %s for network %s" % (runId, networkCreationTime, netId))

        cursor = con.cursor()
        cursor.execute("select s.generation from league_players_snapshot s, networks n where s.network = n.id and n.creation <= %s and n.run = %s order by creation desc limit 1", (networkCreationTime, runId))
        genRows = cursor.fetchall()
        if len(genRows) == 0:
            useGeneration = None 
        else:
            useGeneration = genRows[0][0]
        cursor.close()

        # the very first network did not get any generation, which is at least a bit odd, generations should be faster than iterations.
        if useGeneration is None:
            
            cursor = con.cursor()
            cursor.execute("select min(s.generation) from league_players_snapshot s, networks n where s.network = n.id and n.run = %s", (runId, ))
            minGenRows = cursor.fetchall()
            if len(minGenRows) > 0 and minGenRows[0][0] is not None:
                useGeneration = minGenRows[0][0]
            else:
                useGeneration = 1
            logMsg("falling back to generation %i for network %s" % (useGeneration, netId))
            cursor.close()

        print("useGeneration is", useGeneration)

        cursor = con.cursor()
        cursor.execute("select s.id, s.rating, p.parameter_vals from league_players_snapshot s, networks n, league_players p where s.id = p.id and s.network = n.id and s.generation = %s and n.run = %s order by rating desc", (useGeneration, runId))
        rows = cursor.fetchall()

        players = []

        for row in rows:
            players.append([row[0], row[1], json.loads(row[2])])

        return players

    finally:
        if cursor:
            cursor.close()
        pool.putconn(con)

class BestPlayerResource():

    def __init__(self, pool):
        self.pool = pool

    # runId is not necessary, as networks use UUIDs, so they are unique over the entire database, all runs, anyway.
    def on_get(self, req, resp, net_id):
        players = selectBestPlayersForNetwork(self.pool, net_id)
        if len(players) > 0:
            print("best player for network %s is %s with rating at the time of %i" % (net_id, players[0][0], players[0][1]))
            resp.media = players[0][2]
        else:
            print("There is no best player for network %s" % net_id)
            resp.media = dict()
        resp.status = falcon.HTTP_200


class NetPlayersResource():
    def __init__(self, pool):
        self.pool = pool

    # runId is not necessary, as networks use UUIDs, so they are unique over the entire database, all runs, anyway.
    def on_get(self, req, resp, net_id):
        players = selectBestPlayersForNetwork(self.pool, net_id)
        resp.media = players
        resp.status = falcon.HTTP_200
