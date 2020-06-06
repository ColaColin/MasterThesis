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


class BestPlayerResource():

    def __init__(self, pool):
        self.pool = pool

    # runId is not necessary, as networks use UUIDs, so they are unique over the entire database, all runs, anyway.
    def on_get(self, req, resp, net_id):
        try:
            con = self.pool.getconn()
            cursor = con.cursor()

            cursor.execute("select p.parameter_vals from league_players p inner join (select distinct player1 as pid from league_matches where network = %s union select player2 as pid from league_matches where network = %s) foo on foo.pid = p.id order by p.rating desc limit 1", (net_id, net_id))
            rows = cursor.fetchall()

            if len(rows) == 0:
                resp.media = {}
                resp.status = falcon.HTTP_200
            else:
                resp.media = json.loads(rows[0][0])
                resp.status = falcon.HTTP_200
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)

class NetPlayersResource():
    def __init__(self, pool):
        self.pool = pool

    # runId is not necessary, as networks use UUIDs, so they are unique over the entire database, all runs, anyway.
    def on_get(self, req, resp, net_id):
        try:
            con = self.pool.getconn()
            cursor = con.cursor()

            cursor.execute("select id, rating, parameter_vals from league_players p inner join (select distinct player1 as pid from league_matches where network = %s union select player2 as pid from league_matches where network = %s) foo on foo.pid = p.id order by p.rating desc", (net_id, net_id))
            rows = cursor.fetchall()

            result = []
            for row in rows:
                result.append([row[0], row[1], json.loads(row[2])])
            resp.media = result
            resp.status = falcon.HTTP_200
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)