import threading
import time
from utils.prints import logMsg, setLoggingEnabled
from utils.req import requestBytes, postBytes, requestJson
from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

import sys

import random

import time

import sys

import numpy as np

class EvaluationWorker():
    """
    A worker that polls for packages of game states to be fetched from the evalManager server and 
    pushes policy iterations back to that server.
    """

    def __init__(self, initialState, policy, policyIterator, policyUpdater, isFrameTimeTest = False):
        self.initialState = initialState
        self.policy = policy
        self.policyIterator = policyIterator
        self.policyUpdater = policyUpdater
        self.workQueue = []
        self.resultsQueue = []

        self.initialPolicyID = self.policy.getUUID()

        self.isFrameTimeTest = isFrameTimeTest

        self.command = sys.argv[sys.argv.index("--command")+1].replace("https", "http").replace(":8042", "")
        self.command += ":4242"

        # can be used in case the eval server is somewhere else. That is the case in frametime evaluation.
        if "--evalserver" in sys.argv:
            self.command = sys.argv[sys.argv.index("--evalserver")+1]

        self.workerName = "unknown"
        if "--eval" in sys.argv:
            self.workerName = sys.argv[sys.argv.index("--eval")+1]

        logMsg("Started evaluation worker, talking to eval server on %s" % self.command)

        self.iterateTimes = [1]

        self.lastIterationCompleted = time.monotonic()

        self.printNoWork = True

    def doEvaluation(self, nextWork):
        startTime = time.monotonic()
        games = nextWork["work"]
        workId = nextWork["id"]

        self.lastIterationCompleted = time.monotonic()
        iteratedPolicy = self.policyIterator.iteratePolicy(self.policy, games)
        self.iterateTimes.append(time.monotonic() - self.lastIterationCompleted)

        if len(self.iterateTimes) > 20:
            self.iterateTimes = self.iterateTimes[-20:]

        result = dict()
        result["iterations"] = iteratedPolicy
        if self.initialPolicyID != self.policy.getUUID() and not self.isFrameTimeTest:
            result["network"] = self.policy.getUUID()
        else:
            result["network"] = None
        result["workerName"] = self.workerName

        rpack = dict()
        rpack["id"] = workId
        rpack["data"] = result

        logMsg("Completed work package %s in %.2fs using network %s. Average completion time is now %.2f" % (workId, (time.monotonic() - startTime), result["network"], np.mean(self.iterateTimes)))
        return rpack

    def main(self):
        setLoggingEnabled(True)

        self.pullThread = threading.Thread(target=self.pollWork)
        self.pullThread.daemon = True
        self.pullThread.start()

        self.pushThread = threading.Thread(target=self.pushResults)
        self.pushThread.daemon = True
        self.pushThread.start()

        printNoWork = True

        while True:
            while len(self.workQueue) == 0:
                if printNoWork:
                    logMsg("I have no work!")
                    printNoWork = False
                time.sleep(0.05)

            printNoWork = True

            self.policy = self.policyUpdater.update(self.policy)

            nextWork = self.workQueue[0]
            rpack = self.doEvaluation(nextWork)

            self.resultsQueue.append(rpack)
            del self.workQueue[0]

    def pollWork(self):
        logMsg("Started work poll thread")
        lastSuccess = time.monotonic()

        while True:
             
            while (len(self.workQueue) == 1 and (time.monotonic() - max(lastSuccess, self.lastIterationCompleted)) > np.mean(self.iterateTimes) * 0.8) or len(self.workQueue) > 1:
                time.sleep(0.05)

            #print("wqueue length", len(self.workQueue), (time.monotonic() - max(lastSuccess, self.lastIterationCompleted)) > np.mean(self.iterateTimes) * 0.8)

            workList = requestJson(self.command + "/queue", "")
            if len(workList) > 0:
                pickWork = random.choice(workList[:5])
                try:
                    myWork = requestBytes(self.command + "/checkout/" + pickWork, "", retries=0)
                except:
                    # somebody else took the work before us
                    logMsg("Failed to checkout a task %s" % pickWork)
                    time.sleep(0.3 + random.random() * 0.2)
                    continue

                # decodedWork should be a list of game.store(), so load them via game.load()
                decodedWork = decodeFromBson(myWork)
                games = [self.initialState.load(w) for w in decodedWork]

                self.printNoWork = True

                logMsg("Got work: %i game states" % len(games))
                dwork = dict()
                dwork["work"] = games
                dwork["id"] = pickWork
                self.workQueue.append(dwork)
                lastSuccess = time.monotonic()
            else:
                if self.printNoWork:
                    logMsg("No work found on the server, will keep trying...")
                    self.printNoWork = False
                time.sleep(0.5)

    def pushResults(self):
        logMsg("Started poll results thread")

        while True:
            while len(self.resultsQueue) == 0:
                time.sleep(0.2)
            
            nextResult = self.resultsQueue[0]
            del self.resultsQueue[0]

            resultId = nextResult["id"]
            resultData = encodeToBson(nextResult["data"])
            postBytes(self.command + "/checkin/" + resultId, "", resultData)

                        

                


