import falcon

from core.command.runs import RunsResource
from core.command.reports import ReportsResource
from core.command.login import LoginResource
from core.command.state import StateResource

import psycopg2
from psycopg2 import pool

class AuthMiddleware(object):
    def __init__(self, password):
        self.password = password

    def process_request(self, req, resp):
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

    #app = falcon.API(middleware=[AuthMiddleware(config["secret"])])
    app = falcon.API()

    if config["staticPath"] is not None:
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

        reports = ReportsResource()
        app.add_route("/api/reports/{report_id}", reports)

        state = StateResource(pool, config)
        app.add_route("/api/state/{key}/{entity_id}", state)

        app.add_route("/password", LoginResource(config["secret"]))

        return app

    except (Exception, psycopg2.DatabaseError) as error :
        print ("Error while setting up app", error)
        raise error

