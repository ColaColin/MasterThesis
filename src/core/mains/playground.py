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

# random testing code

if __name__ == "__main__":
    setLoggingEnabled(True)

    def makeReport():
        game = Connect4GameState(7,6,4)
        game = game.playMove(np.random.randint(7))
        game = game.playMove(np.random.randint(7))

        report = dict()
        report["knownResults"] = [1]
        report["generics"] = dict()
        report["policyIterated"] = np.random.rand(7).astype(np.float32)
        report["generics"]["wtf"] = np.random.rand(7).astype(np.float32)
        report["uuid"] = str(uuid.uuid4())
        report["state"] = game.store()

        return report

    report = makeReport()

    # print(report)
    # enc = encodeToBson(report)
    # print("\nEncoded into %i bytes: " % len(enc), enc)
    # print("\nDecoded:\n", decodeFromBson(enc))

    reps = [makeReport() for _ in range(10000)]

    print(reps[0])

    encStart = time.time()
    repEnc = encodeToBson(reps)
    print("Encoding time taken:", time.time() - encStart)

    print("Encoded %i reports into %i kbyte" % (len(reps), 1+(len(repEnc) / 1000)))

    reportId = requests.post(url="http://127.0.0.1:8000/reports/", data=repEnc).json()

    print("Posted report of %i bytes and got response: %s" % (len(repEnc), reportId))

    response = requests.get(url="http://127.0.0.1:8000/reports/" + reportId, stream=True)

    if response.status_code != 200:
        print("Got HTTP error on GET: ", response.status_code, response.content)
        exit(-1)

    redownloaded = response.raw.data

    print("Get report gave us %i bytes" % len(redownloaded))

    decStart = time.time()
    repDec = decodeFromBson(redownloaded)
    print("Decode time taken:", time.time() - decStart)

    print("Decoded %i" % len(repDec))

    print(repDec[0])
    

    # turns out typos in the yaml definition cause confusing errors...

    # mnk = MNKGameState(3, 3, 3)
    # optimizerArgs = dict([("lr", 0.001), ("weight_decay", 0.001)])
    # resnet = PytorchPolicy(128, 1, 16, 1, 1, mnk, "cuda:0", "torch.optim.adamw.AdamW", optimizerArgs)
    # wtfMap0 = dict([("expansions", 15), ("cpuct", 1.5), ("rootNoise", 0.2), ("drawValue", 0.5)])
    # mcts = MctsPolicyIterator(**wtfMap0)
    # tempDecider = TemperatureMoveDecider(12)

    # wtfMap1 = dict([("initalState", mnk), ("policy", resnet), ("policyIterator", mcts), ("gameCount", 128), ("moveDecider", tempDecider)])

    # selfplayer = LinearSelfPlayWorker(**wtfMap1)
    # print(selfplayer)

    # config = mlconfig.load("test.yaml")

    # b = config.instanceB(recursive=True)
    # b.wtf()
