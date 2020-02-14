import falcon

import uuid

from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

import os
from pathlib import Path

def getUUIDPath(uuid, depth = 3):
        """
        to prevent having a single directory with way too many files,
        instead put files into directories by their uuids
        """
        noStrings = uuid.replace("-", "")
        path = ["binary"]
        cnt = 0
        for c in uuid:
            path.append(c)
            cnt += 1;
            if cnt == depth:
                break

        path.append(uuid)

        return os.path.join(*path)

def storeFileUnderPath(path, binaryContent):
    """
    store binary data under path
    """
    dirName = os.path.dirname(path)
    Path(dirName).mkdir(parents=True, exist_ok=True)
    with open(path, "w+b") as f:
        f.write(binaryContent)

def readFileUnderPath(path):
    with open(path, "rb") as f:
        return f.read()

class StateResource():
    def __init__(self, pool, config):
        """
        @param pool: psycopg2 connection pool
        @param config: Configuration plain object
        """
        self.pool = pool
        self.config = config

    def getRelativePath(self, reportUUID, depth = 3):
        return getUUIDPath(reportUUID, depth)

    def getMajorityPolicyUUID(self, decoded):
        """
        in case of a reports package contains states from different networks, just use the most common one.
        slightly inaccurate, but not too important
        """
        results = {}
        for d in decoded:
            if not d["policyUUID"] in results:
                results[d["policyUUID"]] = 1
            else:
                results[d["policyUUID"]] += 1
        biggestPolicy = decoded[0]["policyUUID"]
        biggestPolicyCnt = results[biggestPolicy]
        for k in results:
            if results[k] > biggestPolicyCnt:
                biggestPolicyCnt = results[k]
                biggestPolicy = k
        return biggestPolicy

    def on_get(self, req, resp, key, entity_id):
        mode = key

        if mode == "list":
            # list all reports for the run defined by the entity_id

            try:
                con = self.pool.getconn()
                cursor = con.cursor()
                
                cursor.execute("select id, package_size, worker, creation, iteration, network from states where run = %s", (entity_id, ))
                rows = cursor.fetchall()

                result = []
                for row in rows:
                    result.append({
                        "id": row[0],
                        "packageSize": row[1],
                        "worker": row[2],
                        "creation": int(row[3].timestamp() * 1000),
                        "iteration": row[4],
                        "network": row[5]
                    });

                resp.media = result

                resp.statue = falcon.HTTP_200

            finally:
                if cursor:
                    cursor.close()
                self.pool.putconn(con)

        elif mode == 'download':
            # download a report file by entity_id of the file
            
            try:
                con = self.pool.getconn()
                cursor = con.cursor()

                cursor.execute("select id from states where id = %s", (entity_id, ))
                rows = cursor.fetchall()

                if len(rows) == 0:
                    raise falcon.HTTPError(falcon.HTTP_400, "bad state id")

                fpath = os.path.join(self.config["dataPath"], getUUIDPath(entity_id))

                resp.content_type = "application/octet-stream"

                resp.data = readFileUnderPath(fpath)

                resp.status = falcon.HTTP_200

            finally:
                if cursor:
                    cursor.close()
                self.pool.putconn(con)

        else:
            raise falcon.HTTPError(falcon.HTTP_400, "unknown mode for state api only supporting list or download. Got " + str(mode))

    def on_post(self, req, resp, key, entity_id):
        """
        reports are as files on disk. The database only contains the path of the file, relative
        to the dataPath that is configured in the config file.
        """

        worker_name = key

        newId = str(uuid.uuid4())

        binary = req.bounded_stream.read()

        decoded = decodeFromBson(binary)

        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            
            storeFileUnderPath(os.path.join(self.config["dataPath"], getUUIDPath(newId)), binary)

            packageSize = len(decoded)

            cursor.execute("select iterations, id from runs_info where id = %s", (entity_id, ))
            rows = cursor.fetchall()
            if cursor:
                cursor.close()

            if rows is None or len(rows) != 1:
                raise falcon.HTTPError(falcon.HTTP_400, "Unknown run id")

            iteration = int(rows[0][0])

            networkUUID = self.getMajorityPolicyUUID(decoded)

            if iteration == 0:
                # the random network of iteration 0 is not stored in the database,
                # so don't try to set a foreign key.
                # all other networks should come from the database, so they must exist there
                networkUUID = None

            cursor = con.cursor()

            cursor.execute("insert into states (id, package_size, worker, iteration, network, run) VALUES (%s, %s, %s, %s, %s, %s)",
                (newId, packageSize, worker_name, iteration, networkUUID, entity_id));

            con.commit()
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)


        resp.media = newId

        resp.status = falcon.HTTP_200