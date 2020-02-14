import mlconfig

import mlconfig

from utils.prints import logMsg, setLoggingEnabled

from impls.selfplay.LinearSelfPlay import LinearSelfPlayWorker
from impls.policyIterators.mcts0.mcts0 import MctsPolicyIterator
from impls.selfplay.movedeciders import TemperatureMoveDecider
from impls.games.mnk.mnk import MNKGameState
from impls.games.connect4.connect4 import Connect4GameState
from impls.singleprocess.singleprocess import SingleProcessReporter, SingleProcessUpdater
from impls.polices.pytorch.policy import PytorchPolicy

from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

import bson
import uuid

import numpy as np

import time

import requests

import psycopg2

# random testing code

def makeReport():
    game = Connect4GameState(7,6,4)
    game = game.playMove(np.random.randint(7))
    game = game.playMove(np.random.randint(7))

    report = dict()
    report["knownResults"] = [1]
    report["generics"] = dict()
    report["policyUUID"] = "ab98246f-4b80-48e8-97fc-d365d4a3aa3d"
    report["policyIterated"] = np.random.rand(7).astype(np.float32)
    report["generics"]["wtf"] = np.random.rand(7).astype(np.float32)
    report["uuid"] = str(uuid.uuid4())
    report["state"] = game.store()

    return report


def reportsApiTest2():
    reps = [makeReport() for _ in range(1000)]

    encStart = time.time()
    repEnc = encodeToBson(reps)
    print("Encoding time taken per state:", int(((time.time() - encStart) / len(reps)) * 1000000), "us")

    print("===")

    reportId = requests.post(url="http://127.0.0.1:8042/api/state/test/4492b3fc-0989-497d-b78c-5d97f148bfd4",
        data=repEnc).json()

    print(reportId)
    



def reportsApiTest():

    report = makeReport()

    # print(report)
    # enc = encodeToBson(report)
    # print("\nEncoded into %i bytes: " % len(enc), enc)
    # print("\nDecoded:\n", decodeFromBson(enc))

    reps = [makeReport() for _ in range(1000)]

    print(reps[0])

    encStart = time.time()
    repEnc = encodeToBson(reps)
    print("Encoding time taken:", time.time() - encStart)

    print("Encoded %i reports into %i kbyte" % (len(reps), 1+(len(repEnc) / 1000)))

    reportId = requests.post(url="http://127.0.0.1:8042/api/reports/", data=repEnc).json()

    print("Posted report of %i bytes and got response: %s" % (len(repEnc), reportId))

    response = requests.get(url="http://127.0.0.1:8042/api/reports/" + reportId, stream=True)
    reponse.raise_for_status()

    redownloaded = response.raw.data

    print("Get report gave us %i bytes" % len(redownloaded))

    decStart = time.time()
    repDec = decodeFromBson(redownloaded)
    print("Decode time taken:", time.time() - decStart)

    print("Decoded %i" % len(repDec))

    print(repDec[0])

def postgresTest():
    try:
        connect_str = "dbname='x0' user='x0' host='127.0.0.1' password='x0'"
        # use our connection values to establish a connection
        conn = psycopg2.connect(connect_str)
        # create a psycopg2 cursor that can execute queries
        cursor = conn.cursor()
        # create a new table with a single column called "name"
        cursor.execute("""CREATE TABLE tutorials (name char(40));""")
        # run a SELECT statement - no data in there, but we can try it
        cursor.execute("""SELECT * from tutorials""")
        conn.commit() # <--- makes sure the change is shown in the database
        rows = cursor.fetchall()
        print(rows)
        cursor.close()
        conn.close()
    except Exception as e:
        print("Uh oh, can't connect. Invalid dbname, user or password?")
        print(e)


if __name__ == "__main__":
    setLoggingEnabled(True)

    reportsApiTest2()

    
