import falcon

import uuid

from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

import os
from core.command.state import storeFileUnderPath, getUUIDPath, readFileUnderPath

class NetworksResource():
    def __init__(self, pool, config):
        """
        @param pool: psycopg2 connection pool
        @param config: Configuration plain object
        """
        self.pool = pool
        self.config = config

    def on_post(self, req, resp, param1, param2):
        run_id = param1
        network_id = param2

        binary = req.bounded_stream.read()

        try:
            con = self.pool.getconn()
            cursor = con.cursor()

            cursor.execute("insert into networks (id, run) VALUES (%s, %s)", (network_id, run_id));

            storeFileUnderPath(os.path.join(self.config["dataPath"], getUUIDPath(network_id)), binary)

            con.commit()
        finally:
            if cursor:
                cursor.close()
            con.rollback()
            self.pool.putconn(con)

    def on_get(self, req, resp, param1, param2):
        mode = param1

        if mode == "list":
            run_id = param2
            try:
                con = self.pool.getconn()
                cursor = con.cursor()
                
                cursor.execute("select id, creation, run, acc_rnd, acc_best from networks where run = %s", (run_id, ))
                rows = cursor.fetchall()

                result = []
                for row in rows:
                    result.append({
                        "id": row[0],
                        "creation": int(row[1].timestamp() * 1000),
                        "run": row[2],
                        "accRnd": row[3],
                        "accBest": row[4]
                    })

                resp.media = result

                resp.status = falcon.HTTP_200

            finally:
                if cursor:
                    cursor.close()
                self.pool.putconn(con)

        elif mode == "download":
            network_id = param2

            try:
                con = self.pool.getconn()
                cursor = con.cursor()

                cursor.execute("select id from networks where id = %s", (network_id, ))
                rows = cursor.fetchall()

                if len(rows) == 0:
                    raise falcon.HTTPError(falcon.HTTP_200, "bad network id")
                    
                fpath = os.path.join(self.config["dataPath"], getUUIDPath(network_id))

                resp.content_type = "application/octet-stream"

                resp.data = readFileUnderPath(fpath)

                resp.status = falcon.HTTP_200

            finally:
                if cursor:
                    cursor.close()
                self.pool.putconn(con)

        else:
            raise falcon.HTTPError(falcon.HTTP_400, "bad mode: " + str(mode))