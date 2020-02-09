import falcon

from core.command.runs import RunsResource
from core.command.reports import ReportsResource
from core.command.login import LoginResource

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

    app = falcon.API(middleware=[AuthMiddleware(config["secret"])])

    if config["staticPath"] is not None:
        app.add_static_route("/", config["staticPath"])
        print("Will serve static files from " + config["staticPath"])

    runs = RunsResource()
    app.add_route("/api/runs", runs)
    app.add_route("/api/runs/{run_id}", runs)

    reports = ReportsResource()
    app.add_route("/api/reports/{report_id}", reports)

    app.add_route("/password", LoginResource(config["secret"]))

    return app