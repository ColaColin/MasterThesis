import falcon
import json

class FrametimeResource():
    def __init__(self, pool):
        self.pool = pool

    def on_get(self, req, resp, network_id):

        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute("select run, id from networks where frametime is null order by creation desc");
            rows = cursor.fetchall()

            result = []

            for row in rows:
                result.append({
                    "run": row[0],
                    "network": row[1]
                });

            resp.media = result
            resp.status = falcon.HTTP_200
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)


    def on_post(self, req, resp, network_id):
        result = req.media
        frametime = result["frametime"]
        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute("update networks set frametime = %s where id = %s", (frametime, network_id))
            con.commit()
            resp.status = falcon.HTTP_200
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)