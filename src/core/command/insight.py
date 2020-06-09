import falcon
import json
import os
from utils.misc import readFileUnderPath, constructor_for_class_name

from core.command.state import getUUIDPath
from utils.bsonHelp.bsonHelp import decodeFromBson
import datetime

import multiprocessing as mp

def formatInsightForPath(fpath):
    decoded = decodeFromBson(readFileUnderPath(fpath))

    result = []
    for record in decoded:
        protoState = constructor_for_class_name(record["gameCtor"])(**record["gameParams"])
        gameState = protoState.load(record["state"])

        networkMoves = record["generics"]["net_priors"]
        networkWins = record["generics"]["net_values"]
        iteratedMoves = record["policyIterated"]
        knownResults = record["knownResults"]

        if "reply" in record:
            replyMoves = record["reply"]
        else:
            replyMoves = None

        txt = datetime.datetime.fromtimestamp(record["creation"]).strftime('%Y-%m-%d %H:%M:%S.%f') + "\n"
        txt += "\nPlayed by policy " + record["policyUUID"] + "\n"
        if "numIterations" in record:
            if "remIterations" in record:
                txt += "Clock: " + str(record["numIterations"]) + "/" + str(record["remIterations"]) + "\n"
            else:
                txt += "Number of MCTS iterations: " + str(record["numIterations"]) + "\n"
        txt += gameState.prettyString(networkMoves, networkWins, iteratedMoves, knownResults, replyMoves=replyMoves) + "\n"

        result.append(txt)
    return result

class InsightResource():

    def __init__(self, config):
        self.config = config
        self.pool = mp.Pool(processes=1)

    def on_get(self, req, resp, report_id):

        if report_id == "unknown":
            resp.media = ""
        else:
            fpath = os.path.join(self.config["dataPath"], getUUIDPath(report_id))
            rasync = self.pool.apply_async(formatInsightForPath, (fpath, ))
            resp.media = rasync.get()

        resp.status = falcon.HTTP_200