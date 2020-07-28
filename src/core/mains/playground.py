import mlconfig

import mlconfig

from core.mains.mlconfsetup import loadMlConfig

from utils.prints import logMsg, setLoggingEnabled

from impls.selfplay.LinearSelfPlay import LinearSelfPlayWorker
from impls.selfplay.movedeciders import TemperatureMoveDecider
from impls.games.mnk.mnk import MNKGameState
from impls.games.connect4.connect4 import Connect4GameState
from impls.singleprocess.singleprocess import SingleProcessReporter, SingleProcessUpdater
from impls.polices.pytorch.policy import PytorchPolicy

from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

from core.training.StreamTrainingWorker import SlowWindowSizeManager

from impls.selfplay.TreeSelfPlay import MCTSNode, rolloutPosition, rolloutWithUniformMCTS, tryMCTSRollout

from impls.polices.pytorch.policy import ResCNN
import pytorch_model_summary as pms


import bson
import uuid

import numpy as np

import time

import requests

import psycopg2

from utils.misc import hConcatStrings

from utils.req import postBytes, requestJson, requestBytes

import datetime

import random

import pickle

import os
import abc

import numpy as np

import torch
import torch.cuda
import torch.nn as nn
from torch.autograd import Variable

import uuid

from core.base.Policy import Policy, ExamplePrepareWorker

from utils.encoding import stringToBytes, bytesToString

import pytorch_model_summary as pms

import io

import random

import math

import time

from utils.misc import constructor_for_class_name, IterationCalculatedValue

from utils.prints import logMsg

import sys

from torch.nn.utils.clip_grad import clip_grad_norm_

# random te
# sting code

def saveNetwork(net):
    uid = str(uuid.uuid4())
    uuidBytes = stringToBytes(uid)
    ublen = np.array([uuidBytes.shape[0]], dtype=np.uint8)

    buffer = io.BytesIO()
    torch.save(net.state_dict(), buffer)

    modelBytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    result = encodeToBson(np.concatenate((ublen, uuidBytes, modelBytes)))

    with open(uid + ".network", "w+b") as f:
        f.write(result)

def getRandomGame():
    game = Connect4GameState(7,6,4)
    for _ in range(np.random.randint(25)):
        if game.hasEnded():
            break
        moves = game.getLegalMoves()
        game = game.playMove(random.choice(moves))
    return game

def makeReport():
    game = getRandomGame()

    report = dict()
    report["knownResults"] = [1]
    report["generics"] = dict()
    report["policyUUID"] = "ab98246f-4b80-48e8-97fc-d365d4a3aa3d"
    report["policyIterated"] = np.random.rand(7).astype(np.float32)
    report["generics"]["wtf"] = np.random.rand(7).astype(np.float32)
    report["generics"]["wt2f"] = np.random.rand(7).astype(np.float32)
    report["uuid"] = str(uuid.uuid4())
    report["state"] = game.store()
    report["gameCtor"] = game.getGameConstructorName()
    report["gameParams"] = game.getGameConstructorParams()
    report["gamename"] = game.getGameName()
    report["creation"] = datetime.datetime.utcnow().timestamp()
    report["reply"] = np.random.rand(7).astype(np.float32)
    report["final"] = game.hasEnded()

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
    
def reportsApiTest3():

    policy = PytorchPolicy(32, 1, 32, 3, 64, Connect4GameState(7,6,4), "cuda:0", "torch.optim.adamw.AdamW", {
        "lr": 0.001,
        "weight_decay": 0.0001
    })

    encoded = encodeToBson(policy.store())

    print("Encoded network into " + str(len(encoded)))

    response = requests.post(url="http://127.0.0.1:8042/api/networks/c48b01d7-18e8-4df0-9c8a-d9886473bb49/" + policy.getUUID(),
        data=encoded)

    response.raise_for_status()
    
    response = requests.get(url="http://127.0.0.1:8042/api/networks/download/" + policy.getUUID(), stream=True)
    response.raise_for_status()

    redownloaded = decodeFromBson(response.raw.data)

    prevId = policy.getUUID()

    policy.reset()

    policy.load(redownloaded)

    print(policy.getUUID(), prevId)

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

import torch

if __name__ == "__main__":
    setLoggingEnabled(True)

    # game = Connect4GameState(7,6,4)
    # game = game.playMove(0)
    # game = game.playMove(6)
    # game = game.playMove(1)
    # game = game.playMove(5)
    # game = game.playMove(2)

    # start = time.monotonic_ns()
    # x = rolloutPosition(game, 500)
    # end = time.monotonic_ns()

    # rtime = (end - start) / 1000000.0

    # print(rtime, x)

    net = ResCNN(6, 7, 1, 3, 128, 32, 3, 7, 3, 1, mode="sq")
    print(pms.summary(net, torch.zeros((1, 1, 6, 7))))
    saveNetwork(net)


    # genStart = time.monotonic()
    # package = [makeReport() for _ in range(2048)]
    # genFinished = time.monotonic()

    # print("Generated package in %.4f" % (genFinished - genStart))

    # encStart = time.monotonic()
    # encoded = pickle.dump(package, open("save.p", "wb"))
    # encFinished = time.monotonic()

    # print("Encoded package in %.4f" % (encFinished - encStart))

    # foobar = encodeToBson(np.random.dirichlet([0.9] * 3))
    # print(foobar)

    # game = getRandomGame()

    # node = MCTSNode(game)
    # node.nodes[42] = "foo"

    # node2 = MCTSNode(game)
    # # node2.nodes[42] = "bar"
    # print(node.nodes[42])
    # print(node2.nodes)

    # game = getRandomGame()

    # print(str(game))

    # report = game.store()
    # encReport = encodeToBson(report)

    # resp = postBytes("http://127.0.0.1:4242/queue/", "", encReport, expectResponse=True)

    # workList = requestJson("http://127.0.0.1:4242/queue", "")
    # # print("Work on server is", workList)

    # myWork = requestBytes("http://127.0.0.1:4242/checkout/" + resp, "")

    # print(str(game.load(decodeFromBson(myWork))))

    # postBytes("http://127.0.0.1:4242/checkin/" + resp, "", encReport)

    # workResults = requestJson("http://127.0.0.1:4242/results", "")

    # wResultBytes = requestBytes("http://127.0.0.1:4242/results/" + workResults[0], "")

    # print(str(game.load(decodeFromBson(wResultBytes))))

    # reportsApiTest3()

    # wm = SlowWindowSizeManager(550000, 5, 15, 2000000, 180000, 8096)

    # for i in range(20):
    #     print("i = %i, Window size: %i, iteration size: %i" % (i, wm.getWindowSize(i), wm.getIterationSize(i)))

    # games = [str(getRandomGame()) for _ in range(3)]
    # games[1] += "\nOne more line"
    
    # print(hConcatStrings(games))

    # core = loadMlConfig("confs/distributedworker.yaml")
    # league = core.serverLeague()

    # p1 = league.newPlayer()

    # p1M1 = league.mutatePlayer(p1)
    # p1M2 = league.mutatePlayer(p1)

    # print(p1)
    # print(p1M1)
    # print(p1M2)

    # print(league.getNewRatings(1700, 1600, 0.5))