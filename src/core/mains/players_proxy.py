# small program which runs on localhost:1337 and provides a single call to get the players list by run
# -> on startup tries to bind to localhost:1337, exits if that fails
# -> first request for a run id causes it to start polling that run in some regular interval and caching it
# -> shuts down after 60 seconds without anybody doing a request
# -> can be started with extra parameter to filter players to a specific policy UUID for the frametime evaluator
# -> requires --secret and --command parameters to know where to connect to

# the reason for the existance of this program is to reduce the command server load, as it means not every worker process hammers
# the server, just one instance of this proxy per worker-machine.

import falcon
from utils.req import requestJson, postJson
import random

import threading
import time
import setproctitle

from utils.prints import logMsg, setLoggingEnabled

from wsgiref.simple_server import make_server
import sys

import os

import subprocess

def tryPlayersProxyProcess(command, secret, forPolicy=None):
    cmd = ["python", "-m", "core.mains.players_proxy", "--command", command, "--secret", secret]
    if forPolicy is not None:
        cmd += ["--policy", forPolicy]
    return subprocess.Popen(cmd)

class ProxyResource():
    def __init__(self, command, secret, forPolicy):
        self.cached = dict()
        self.command = command
        self.secret = secret
        self.forPolicy = forPolicy
        self.pendingReports = dict()
        self.lastDataRequest = time.monotonic()
        pollThread = threading.Thread(target=self.pollData)
        pollThread.daemon = True
        pollThread.start()

    def pollData(self):
        logMsg("Start polling thread player_proxy!")
        while True:
            for runId in self.pendingReports:
                pr = []
                prSrc = self.pendingReports[runId]
                while len(prSrc) > 0:
                    pr.append(prSrc.pop())
                if len(pr) > 0:
                    self.postReports(runId, pr)

            for runId in self.cached:
                self.cached[runId] = self.queryPlayerList(runId)
            
            if time.monotonic() - self.lastDataRequest > 60:
                logMsg("Exit players_proxy, nobody is calling it!")
                os._exit(0)
            
            time.sleep(2 + random.random() * 1)

    def postReports(self, runId, reports):
        if self.forPolicy is None:
            logMsg("Posting %i game results" % len(reports))
            postJson(self.command + "/api/league/reports/" + runId, self.secret, reports)

    def queryPlayerList(self, runId):
        if self.forPolicy is None:
            return requestJson(self.command + "/api/league/players/" + runId, self.secret)
        else:
            return requestJson(self.command + "/api/netplayers/"+self.forPolicy, self.secret)
    
    def on_post(self, req, resp, runId):
        self.lastDataRequest = time.monotonic() 
        if not runId in self.pendingReports:
            self.pendingReports[runId] = []
        for rm in req.media:
            self.pendingReports[runId].append(rm)
        resp.status = falcon.HTTP_200

    def on_get(self, req, resp, runId):
        self.lastDataRequest = time.monotonic() 
        if not runId in self.cached:
            self.cached[runId] = self.queryPlayerList(runId)
        logMsg("players_proxy responding with %i players" % len(self.cached[runId]))
        resp.media = self.cached[runId]
        resp.status = falcon.HTTP_200

def startProxy(secret, command, forPolicy=None):
    setproctitle.setproctitle("x0_players_proxy")
    setLoggingEnabled(True)

    app = falcon.API()
    app.add_route("/players/{runId}", ProxyResource(command, secret, forPolicy))

    try:
        with make_server("127.0.0.1", 1337, app) as httpd:
            if forPolicy is None:
                logMsg("Started players_proxy!")
            else:
                logMsg("Started players_proxy for policy %s!" % forPolicy)
            httpd.serve_forever()
    except OSError as error:
        if error.errno == 98: #port in use
            logMsg("Failed to start players_proxy, seems there is already one running!")
        else:
            raise error
    
if __name__ == "__main__":
    hasArgs = ("--secret" in sys.argv) and ("--command" in sys.argv)

    if not hasArgs:
        raise Exception("You need to provide arguments for the players proxy: --secret <server password> and --command <command server host>!")

    secret = sys.argv[sys.argv.index("--secret")+1]
    commandHost = sys.argv[sys.argv.index("--command")+1]
    forPolicy = None
    if "--policy" in sys.argv:
        forPolicy = sys.argv[sys.argv.index("--policy")+1]

    startProxy(secret, commandHost, forPolicy=forPolicy)