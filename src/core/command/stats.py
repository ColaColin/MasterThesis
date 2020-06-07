import falcon
import json

class StatsResource():

    def __init__(self, pool):
        self.pool = pool

    def on_get(self, req, resp, run_id):
        if run_id is None:
            raise falcon.HTTPError(falcon.HTTP_400, "Missing run id")

        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute("SELECT iteration, played_states, new_states, first_player_wins, draws, game_length_avg, game_length_std, avg_nodes from run_iteration_stats where run = %s", (run_id, ));
            rows = cursor.fetchall();

            result = []

            for row in rows:
                result.append({
                    "iteration": row[0],
                    "played_states": row[1],
                    "new_states": row[2],
                    "first_player_wins": row[3],
                    "draws": row[4],
                    "game_length_avg": row[5],
                    "game_length_std": row[6],
                    "avg_nodes": row[7]
                });
            
            resp.media = result

            resp.status = falcon.HTTP_200
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)