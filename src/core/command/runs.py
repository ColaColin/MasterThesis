import falcon
import json
import uuid

class RunsResource():

    def __init__(self, pool):
        self.pool = pool

    def loadRuns(self, id = None):
        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            if id is not None:
                cursor.execute("SELECT id, name, config, creation from runs where id = %s", (id, ));
            else:
                cursor.execute("SELECT id, name, config, creation from runs");
            rows = cursor.fetchall();
            
            result = [];
            for row in rows:
                result.append({
                    "id": row[0],
                    "name": row[1],
                    "config": row[2],
                    "timestamp": int(row[3].timestamp() * 1000)
                })

            return result
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)

    def deleteRun(self, id):
        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute("delete from runs where id = %s", (id, ))
            con.commit()
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)

    def insertRun(self, run):
        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute("insert into runs (id, name, config) VALUES (%s, %s, %s)", (run["id"], run["name"], run["config"]))
            con.commit()
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)

    def parseRun(self, message):
        isRunMsg = "name" in message and "config" in message \
            and isinstance(message["name"], str) and isinstance(message["config"], str)

        if not isRunMsg:
            raise falcon.HTTPError(falcon.HTTP_400, "Malformed run configuration")

        newId = str(uuid.uuid4())

        newRuns = dict()
        newRuns["id"] = newId
        newRuns["name"] = message["name"]
        newRuns["config"] = message["config"]
  
        return newRuns

    def on_delete(self, req, resp, run_id = None):
        if run_id is not None:
            self.deleteRun(run_id)

        resp.status = falcon.HTTP_200;

    def on_get(self, req, resp, run_id = None):
        if run_id is not None:
            selectedRunLst = self.loadRuns(run_id)
            if len(selectedRunLst) == 0:
                raise falcon.HTTPError(falcon.HTTP_404)
            else:
                resp.media = selectedRunLst[0]
        else:
            resp.media = self.loadRuns()

        resp.status = falcon.HTTP_200

    def on_post(self, req, resp, run_id = None):
        """
        Add a run
        """
        message = req.media

        newRun = self.parseRun(message)

        self.insertRun(newRun)

        resp.body = json.dumps(newRun["id"])
        resp.status = falcon.HTTP_200

