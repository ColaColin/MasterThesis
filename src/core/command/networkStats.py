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
            cursor.execute("select run, id from networks where acc_mcts_moves is null order by creation desc");
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

        acc_network_moves = results["acc_network_moves"]
        acc_network_wins = results["acc_network_wins"]
        acc_mcts_moves = results["acc_mcts_moves"]

        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute("update networks set acc_network_moves = %s, acc_network_wins = %s, acc_mcts_moves = %s where id = %s", (acc_network_moves, acc_network_wins, acc_mcts_moves, network_id));
            con.commit()
            resp.status = falcon.HTTP_200
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)