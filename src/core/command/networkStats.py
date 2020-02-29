import falcon
import json

class NetworkStatsResource():
    def __init__(self, pool):
        self.pool = pool

    def on_get(self, req, resp, network_id):
        """
        Returns {run: uuid, network: uuid}[] for networks that still need evaluation
        """

        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute("select run, id from networks where acc_rnd_limited is null order by creation desc");
            rows = cursor.fetchall();

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
        results = req.media
        rnd_limited = results["acc_rnd_limited"]
        best_limited = results["acc_best_limited"]
        rnd_full = results["acc_rnd_full"]
        best_full = results["acc_best_full"]

        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute("update networks set acc_rnd_limited = %s, acc_best_limited = %s, acc_rnd_full = %s, acc_best_full = %s where id = %s", (rnd_limited, best_limited, rnd_full, best_full, network_id));
            con.commit()
            resp.status = falcon.HTTP_200
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)