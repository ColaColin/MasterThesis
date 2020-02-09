import falcon
import json
import uuid

class RunsResource():

    def __init__(self):
        self.activeRuns = [{
            "id": str(uuid.uuid4()),
            "name": "Some test run",
            "config": "yaml file content...",
            "timestamp": 0
        }, {
            "id": str(uuid.uuid4()),
            "name": "Some test run 2",
            "config": "yaml file content...",
            "timestamp": 1
        }]

    def parseRun(self, message):
        isRunMsg = "name" in message and "config" in message \
            and isinstance(message["name"], str) and isinstance(message["config"], str) \
            and "timestamp" in message and isinstance(message["timestamp"], int)

        if not isRunMsg:
            raise falcon.HTTPError(falcon.HTTP_400, "Malformed run configuration")

        newId = str(uuid.uuid4())

        newRuns = dict()
        newRuns["id"] = newId
        newRuns["name"] = message["name"]
        newRuns["config"] = message["config"]
        newRuns["timestamp"] = message["timestamp"]

        return newRuns

    def on_delete(self, req, resp, run_id = None):
        if run_id is not None:
            self.activeRuns = list(filter(lambda r: r["id"] != run_id, self.activeRuns))
            
        resp.status = falcon.HTTP_200;

    def on_get(self, req, resp, run_id = None):
        if run_id is not None:
            selectedRunLst = list(filter(lambda x: x["id"] == run_id, self.activeRuns))
            if len(selectedRunLst) == 0:
                raise falcon.HTTPError(falcon.HTTP_404)
            else:
                resp.media = selectedRunLst[0]
        else:
            resp.media = self.activeRuns

        resp.status = falcon.HTTP_200

    def on_post(self, req, resp, run_id = None):
        """
        Add a run
        """
        message = req.media

        newRuns = self.parseRun(message)

        self.activeRuns.append(newRuns)

        resp.body = json.dumps(newRuns["id"])
        resp.status = falcon.HTTP_200


    def on_put(self, req, resp, run_id = None):
        """
        Update a run
        """

        message = req.media

        newRuns = self.parseRun(message)

        if not "id" in message:
            raise falcon.HTTPError(falcon.HTTP_400, "Missing id to update run")

        newRuns["id"] = message["id"]

        self.activeRuns = list(map(lambda x: newRuns if x["id"] == newRuns["id"] else x, self.activeRuns))

        resp.body = json.dumps(newRuns["id"])
        resp.status = falcon.HTTP_200
