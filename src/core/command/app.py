import falcon

from core.command.runs import RunsResource
from core.command.login import LoginResource
from core.command.state import StateResource
from core.command.networks import NetworksResource
from core.command.stats import StatsResource
from core.command.networkStats import NetworkStatsResource
from core.command.runsha import RunShaResource

import psycopg2
from psycopg2 import pool

class AuthMiddleware(object):
    def __init__(self, password):
        self.password = password

    def process_request(self, req, resp):
        if req.path.startswith("/api/networks/download/"):
            return
        if "api/" in req.path:
            secret = req.get_header("secret")
            if secret != self.password:
                raise falcon.HTTPUnauthorized("You are not supposed to be here")

def defineApp(config):
    """
    Config has keys:
    - staticPath: if not none serve static files from here
    - ... more to come?
    """

    app = falcon.API(middleware=[AuthMiddleware(config["secret"])])
    #app = falcon.API()

    if "staticPath" in config:
        app.add_static_route("/", config["staticPath"])
        print("Will serve static files from " + config["staticPath"])

    try:
        pool = psycopg2.pool.SimpleConnectionPool(1, 20,user = config["dbuser"],
                                              password = config["dbpassword"],
                                              host = "127.0.0.1",
                                              port = "5432",
                                              database = config["dbname"]);

        

        runs = RunsResource(pool)
        app.add_route("/api/runs", runs)
        app.add_route("/api/runs/{run_id}", runs)

        state = StateResource(pool, config)
        app.add_route("/api/state/{key}/{entity_id}", state)

        networks = NetworksResource(pool, config)
        app.add_route("/api/networks/{param1}/{param2}", networks)

        stats = StatsResource(pool)
        app.add_route("/api/stats/{run_id}", stats)

        networkStats = NetworkStatsResource(pool)
        app.add_route("/api/evaluations/{network_id}", networkStats)

        runsha = RunShaResource(pool)
        app.add_route("/sha/{run_id}", runsha)

        app.add_route("/password", LoginResource(config["secret"]))

        return app

    except (Exception, psycopg2.DatabaseError) as error :
        print ("Error while setting up app", error)
        raise error

