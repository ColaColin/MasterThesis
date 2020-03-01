import falcon
import json

class RunShaResource():
    def __init__(self, pool):
        self.pool = pool

    def on_get(self, req, resp, run_id):
        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute("SELECT sha from runs where id = %s", (run_id, ));
            rows = cursor.fetchall();
            
            if len(rows) > 0:
                resp.body = rows[0][0]
            else:
                resp.body = "unknown"

        except Exception as error:
            print("Could not get sha for run", run_id, error)
            resp.body = "unknown"
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)

        resp.content_type = falcon.MEDIA_TEXT
        resp.status = falcon.HTTP_200
