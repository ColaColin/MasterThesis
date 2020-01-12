import falcon

import uuid

from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

class ReportsResource():
    def __init__(self):
        self.knownReports = dict()

    def on_get(self, req, resp, report_id):
        if report_id is None:
            raise falcon.HTTPError(falcon.HTTP_404, "missing report id")
        if not report_id in self.knownReports:
            raise falcon.HTTPError(falcon.HTTP_404, "unknown report id")

        report = self.knownReports[report_id]

        resp.content_type = "application/octet-stream"

        result = encodeToBson(report)

        resp.data = result

        resp.status = falcon.HTTP_200
        

    def on_post(self, req, resp, report_id):
        newId = str(uuid.uuid4())

        decoded = decodeFromBson(req.stream.read())

        self.knownReports[newId] = decoded

        resp.media = newId

        resp.status = falcon.HTTP_200