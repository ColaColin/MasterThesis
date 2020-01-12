import os
import falcon
import json

import sys

from core.command.runs import RunsResource
from core.command.reports import ReportsResource

development = "--reload" in sys.argv

print("Development mode:", development)

api = application = falcon.API()

if development:
    staticPath = os.path.join(os.getcwd(), "core/command/frontend/")
    api.add_static_route("/", staticPath)

runs = RunsResource()
api.add_route("/runs/{run_id}", runs)

reports = ReportsResource()
api.add_route("/reports/{report_id}", reports)