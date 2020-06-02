import falcon
import json
import uuid

from core.mains.mlconfsetup import loadMlConfig
from core.command.runs import RunsResource
import tempfile

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
            self.cachedLeagues[runId] = core.serverLeague()
            return self.cachedLeagues[runId]

    def on_get(self, req, resp, mode, run_id):
        league = self.loadLeague(run_id)

        if mode == "players":
            resp.media = league.getPlayers(self.pool, run_id)
        else:
            resp.media = league.getMatchHistory(self.pool, run_id)

        resp.status = falcon.HTTP_200

    def on_post(self, req, resp, mode = None, run_id = None):
        assert run_id is not None
        league = self.loadLeague(run_id)
        report = req.media
        print("Reported a result", report)
        league.reportResult(report["p1"], report["p2"], report["winner"], run_id, self.pool)

        resp.status = falcon.HTTP_200

