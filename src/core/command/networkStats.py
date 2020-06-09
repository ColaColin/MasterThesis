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

            result = []

            cursor = con.cursor()
            cursor.execute("select n.run, n.id, max(s.generation), n.acc_mcts_moves from networks n left outer join league_players_snapshot s on n.id = s.network group by n.run, n.id, n.acc_mcts_moves order by n.creation desc")
            candiRows = cursor.fetchall()
            cursor.close()

            candidates = []
            lastRun = None
            lastGeneration = None
            for crow in candiRows:
                if crow[0] == lastRun and crow[2] is None:
                    gen = lastGeneration
                else:
                    gen = crow[2]
                if crow[3] is None:
                    candidates.append([crow[0], crow[1], gen])
                    print(candidates[-1])

                if crow[2] is not None:
                    lastRun = crow[0]
                    lastGeneration = crow[2]

            # return only run/network combinations for which:
            #    1. there are no league_players at all for that run
            for runId, networkId, maxG in candidates:
                cursor = con.cursor()
                cursor.execute("select count(*) from league_players where run = %s", (runId, ))
                if cursor.fetchall()[0][0] == 0:
                    result.append({
                        "run": runId,
                        "network": networkId
                    })
                    break
                cursor.close()

                # or 2. there is a generation logged for a newer network in that run
                if maxG is None:
                    continue
                cursor = con.cursor()
                cursor.execute("select max(s.generation) from league_players_snapshot s, networks n where s.network = n.id and n.run = %s", (runId, ));
                mgrows = cursor.fetchall()
                if len(mgrows) > 0:
                    runMaxGeneration = mgrows[0][0];
                    if runMaxGeneration > maxG:
                        result.append({
                            "run": runId,
                            "network": networkId
                        })
                        break

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