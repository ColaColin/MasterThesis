import time

from utils.prints import logMsg, setLoggingEnabled

from utils.bsonHelp.bsonHelp import decodeFromBson

import abc

import random

import numpy as np

import torch.multiprocessing as mp
import threading

from multiprocessing import Queue as PlainQueue

from core.training.NetworkApi import NetworkApi
from utils.bsonHelp.bsonHelp import decodeFromBson, encodeToBson

import time

from utils.req import requestJson, requestBytes

import sys

import traceback

from utils.stacktracing import trace_start

# this better version of the StreamTrainingWorker2 also uses the WindowSizeManager defined in StreamTrainingWorker.py

# ulimit -n 300000 might be needed. Or maybe not anymore.

import collections

def manageStreamWork(trainQueue, eventsQueue, command, secret, run, windowManager, batchSize, examplesBatcher, deduplicate, deduplicationWeight):
    """
        do the management loop that checks how many states are known and controls the iterations.
        this includes checking if training can start. It can only start if all states for the current iteration have been downloaded
        so if on startup it shows there is a full window of examples on the server, wait until that windows has been downloaded.
        
        trainQueue receives ready-made batches of training data and requests for the network, 
        eventsQueue produces notification of completed training batches and network snapshots to be published.
    """

    try:
        setLoggingEnabled(True)
        manager = StreamManagement(command, secret, run, trainQueue, eventsQueue, windowManager, batchSize, examplesBatcher, deduplicate, deduplicationWeight)

        downloader = threading.Thread(target=lambda x: x.doStatesDownloading(), args=(manager, ))
        downloader.start()
        logMsg("Started downloader thread")

        unpacker = threading.Thread(target=lambda x: x.doProcessDownloades(), args=(manager, ))
        unpacker.start()
        logMsg("Started unpacker thread")

        eventHandler = threading.Thread(target=lambda x: x.doHandleTrainingEvents(), args=(manager, ))
        eventHandler.start()
        logMsg("Started event handler thread")

        manager.manage()

    except:
        print("MANAGER ERROR", "".join(traceback.format_exception(*sys.exc_info())))

def prepareFunc(x):
    setLoggingEnabled(True)
    return x[0].prepareExample(x[1])

def blockGet(d):
    while(len(d) == 0):
        time.sleep(0.1)
    return d.popleft()

class StreamManagement():
    def __init__(self, command, secret, run, trainQueue, eventsQueue, windowManager, batchSize, examplesBatcher, deduplicate, deduplicationWeight):
        self.command = command
        self.secret = secret
        self.run = run
        self.trainQueue = trainQueue
        self.eventsQueue = eventsQueue
        self.windowManager = windowManager
        self.deduplicate = deduplicate
        self.deduplicationWeight = deduplicationWeight

        if self.deduplicate:
            logMsg("Deduplication is enabled with a deduplicationWeight of %.2f" % self.deduplicationWeight)
        else:
            logMsg("Deduplication is not enabled!")
            

        self.networks = NetworkApi()

        # the downloading process puts newly downloaded packages after processing into this queue
        self.newPackages = collections.deque()

        # the thread which processes downloaded packages puts the resulting examples into this queue
        self.newQueue = collections.deque()
        
        # pool of game state records that can be picked from for the next mini batch
        # gets filled by newly generated states and drained by the training of the network
        # at a factor that results in it being empty once enough new game states have been generated 
        # for a full network iteration
        self.windowBuffer = []

        self.pullBuffer = []
        self.newBuffer = collections.deque()

        # waiting room for game states that have already been learned from, but might be
        # reused in the next iteration. Gets drained back into the windowBuffer,
        # when the current network iteration is finished
        # states will be prioritized by age, if the windowBuffer has a max size,
        # older states will not be put back and instead dropped entirely.
        self.waitBuffer = []

        # dict of hash -> list of states that have that hash. All states are inserted here and get updated in here for deduplication.
        # states are lists: [prepareExample(), creationTimeStamp, seenCount]
        self.stateRepository = dict()
        self.repositorySize = 0

        # how big are the batches send to the training process
        self.batchSize = batchSize

        self.examplesBatcher = examplesBatcher

        self.pendingCounter = 1
        self.currentMoveLoss = []
        self.currentWinLoss = []

        self.iterationStartTime = 0

        self.trainFrameCount = 0
        self.publishWait = False

        self.serverStates = []
        self.downloadedStates = set()

        self.downloadedStatesCount = 0
        self.pendingDownloads = 0
        self.pendingUnpacks = 0

        self.dedupeFramesUsed = 0

    def acceptNewExample(self, newExample):
        if not self.deduplicate:
            self.newQueue.append(newExample)
        else:
            eHash = self.examplesBatcher.getHashForExample(newExample[0])
            knownAlready = False
            if eHash in self.stateRepository:
                qLst = self.stateRepository[eHash]
                for ql in qLst:
                    if self.examplesBatcher.areExamplesEqual(ql[0], newExample[0]):
                        knownAlready = True
                        self.examplesBatcher.mergeInto(ql[0], newExample[0], self.deduplicationWeight)
                        ql[1] = newExample[1]
                        ql[2] += 1
                        break
                if not knownAlready:
                    qLst.append(newExample)
            else:
                self.stateRepository[eHash] = [newExample]
            
            if not knownAlready:
                self.repositorySize += 1
                self.newQueue.append(newExample)

    def doProcessDownloades(self):
        displayUniqueStatsEvery = 100000
        lastRepoSize = self.repositorySize
        lastDownloadedSize = self.downloadedStatesCount

        try:
            while True:
                nextPackage = blockGet(self.newPackages)
                startDecode = time.monotonic_ns()
                statesList = decodeFromBson(nextPackage)
                endDecode = time.monotonic_ns()

                startPrepare = time.monotonic_ns()
                unpacked = list(map(lambda x: [self.examplesBatcher.prepareExample(x), x["creation"], 1], statesList))
                endPrepare = time.monotonic_ns()

                #logMsg("Completed processing for a downloaded package %.1fms to decode, %.1fms to prepare" % ((endDecode - startDecode) / 1000000, (endPrepare - startPrepare) / 1000000))

                for up in unpacked:
                    self.downloadedStatesCount += 1
                    self.pendingUnpacks -= 1
                    self.acceptNewExample(up)

                    if self.downloadedStatesCount % displayUniqueStatsEvery == 0 and self.deduplicate:
                        repoAdd = self.repositorySize - lastRepoSize
                        downloadAdd = self.downloadedStatesCount - lastDownloadedSize
                        logMsg("Amount of unique frames produced overall: %.2f%%, in last %ik: %.2f%%" % (100.0 * (self.repositorySize / self.downloadedStatesCount), displayUniqueStatsEvery // 1000, 100.0 * (repoAdd / downloadAdd)))
                        lastRepoSize = self.repositorySize
                        lastDownloadedSize = self.downloadedStatesCount

        except:
            print("doProcessDownloades ERROR", "".join(traceback.format_exception(*sys.exc_info())))

    def doStatesDownloading(self):
        try:
            while True:
                self.serverStates = requestJson(self.command + "/api/state/list/" + self.run, self.secret)
                # download oldest statest first, so states are learnt from in order of creation
                self.serverStates.sort(key=lambda x: x["creation"])

                self.pendingDownloads = 0
                for state in self.serverStates:
                    if not state["id"] in self.downloadedStates:
                        self.pendingDownloads += state["packageSize"]

                time.sleep(0.25)

                for state in self.serverStates:
                    if not state["id"] in self.downloadedStates:
                        self.newPackages.append(requestBytes(self.command + "/api/state/download/" + state["id"], self.secret))
                        self.pendingUnpacks += state["packageSize"]
                        self.pendingDownloads -= state["packageSize"]
                        self.downloadedStates.add(state["id"])
                        #logMsg("Downloaded state %s" % state["id"])
        except:
            print("doProcessDownloades ERROR", "".join(traceback.format_exception(*sys.exc_info())))

    def doHandleTrainingEvents(self):
        try:
            while True:
                nextEvent = self.eventsQueue.get()

                if nextEvent[0] == "finish":
                    batchId = nextEvent[1]
                    #logMsg("Finished batch %i" % batchId)
                    self.currentMoveLoss += nextEvent[2]
                    self.currentWinLoss += nextEvent[3]

                elif nextEvent[0] == "network":
                    # publish the network, which finishes the iteration
                    iteration = nextEvent[1]

                    newUUID = nextEvent[2]
                    policyData = nextEvent[3]

                    logMsg("Iteration %i completed, frames processed: %i, of which were seen before: %i" % (iteration, self.trainFrameCount, self.dedupeFramesUsed))
                    logMsg("Iteration network loss: %.4f on moves, %.4f on outcome" % (np.mean(self.currentMoveLoss), np.mean(self.currentWinLoss)))
                    self.networks.uploadEncodedNetwork(newUUID, policyData)

                    self.clearWaitBuffer(iteration)

                    # finally, allow the next iteration to start
                    self.publishWait = False
        except:
            print("doProcessDownloades ERROR", "".join(traceback.format_exception(*sys.exc_info())))

    def clearWaitBuffer(self, iteration):
        cpyc = 0

        logMsg("Draining %i unprocessed new frames into the wait buffer, this number should stay low!" % (len(self.newBuffer) + len(self.pullBuffer)))

        self.waitBuffer += self.newBuffer
        self.waitBuffer += self.pullBuffer

        self.newBuffer = collections.deque()
        self.pullBuffer = []

        self.waitBuffer.sort(key = lambda x: x[1], reverse=True)
        nextWindowSize = self.windowManager.getWindowSize(iteration+1) - self.windowManager.getIterationSize(iteration)
        for frame in self.waitBuffer:
            if len(self.windowBuffer) < nextWindowSize:
                cpyc += 1
                self.windowBuffer.append(frame)
            else:
                break
        dropped = len(self.waitBuffer) - cpyc
        self.waitBuffer = []

        random.shuffle(self.windowBuffer)

        iterationEnd = time.monotonic()
        logMsg("Moved %i states back into the window from the waiting buffer. Dropped %i old ones. Iteration finished, iteration time: %.2fs" % (cpyc, dropped, iterationEnd - self.iterationStartTime))


    def getCurrentIterationNumber(self):
        return len(self.networks.getNetworkList()) + 1

    def getExpectedIterationWindowSize(self, iterationNumber):
        result = 0
        for ix in range(1, iterationNumber+1):
            result += self.windowManager.getIterationSize(ix)
        return min(result, self.windowManager.getWindowSize(iterationNumber))

    def pickFramesForTraining(self, num):
        picked = []
        while len(picked) < num:
            allLen = len(self.windowBuffer) + len(self.newBuffer)
            propNew = 1 - (len(self.newBuffer) / allLen)
            if random.random() > propNew:
                # try to add as much randomness into the order of examples as possible
                if random.random() > 0.5:
                    picked.append(self.newBuffer.pop())
                else:
                    picked.append(self.newBuffer.popleft())
            else:
                picked.append(self.windowBuffer.pop())

        self.waitBuffer += picked

        for pick in picked:
            if pick[2] > 1:
                self.dedupeFramesUsed += 1

        return list(map(lambda x: x[0], picked))

    def drainPullBuffer(self):
        # new examples are first put into the pullBuffer and once the next batch to learn from is ready that buffer is shuffeld.
        random.shuffle(self.pullBuffer)
        if random.random() > 0.5:
            self.newBuffer.extend(self.pullBuffer)
        else:
            self.newBuffer.extendleft(self.pullBuffer)
        self.pullBuffer = []

    def manageLoop(self):
        while True:
            self.iterationStartTime = time.monotonic()
            self.currentMoveLoss = []
            self.currentWinLoss = []
            self.trainFrameCount = 0
            self.dedupeFramesUsed = 0

            iterationNumber = self.getCurrentIterationNumber()
            iterationSize = self.windowManager.getIterationSize(iterationNumber)
            
            oldDataSize = len(self.windowBuffer)
            sumStatesCount = iterationSize + oldDataSize
            framesPerNewFrame = sumStatesCount / iterationSize

            logMsg("Beginning network iteration %i" % iterationNumber)
            logMsg("We have %i old states, want to fit on %i states, this requires %.2f examples learnt per new state" % (oldDataSize, sumStatesCount, framesPerNewFrame))
            newWaiting = len(self.newQueue)
            logMsg("Pending new frames waiting to be processed: %i" % newWaiting)
            logMsg("Pending frames to be downloaded: %i" % self.pendingDownloads)
            logMsg("Pending packages to be unpacked: %i" % self.pendingUnpacks)
            overhang = ((self.pendingDownloads + newWaiting + self.pendingUnpacks) / iterationSize) * 100
            logMsg("That means overhang from the last iteration is %.2f%%" % overhang)
            if overhang > 50:
                logMsg("!!!!!!!!!!!!!!!! Large overhang detected, training is falling behind !!!!!!!!!!!!!!!!")
                logMsg("!!!!!!!!!!!!!!!! Large overhang detected, training is falling behind !!!!!!!!!!!!!!!!")
                logMsg("!!!!!!!!!!!!!!!! Large overhang detected, training is falling behind !!!!!!!!!!!!!!!!")

            pendingForTraining = 0

            lastPrint = time.monotonic() - 999

            for frameNumber in range(iterationSize):
                self.pullBuffer.append(blockGet(self.newQueue))
                pendingForTraining += framesPerNewFrame

                if pendingForTraining >= self.batchSize:
                    self.drainPullBuffer()

                while pendingForTraining >= self.batchSize:
                    nextBatch = self.pickFramesForTraining(self.batchSize)
                    pendingForTraining -= self.batchSize
                    nextBatchPrepared = self.examplesBatcher.packageExamplesBatch(nextBatch)
                    trainId = self.pendingCounter
                    self.pendingCounter += 1
                    #logMsg("Queue batch %i" % trainId)
                    self.trainFrameCount += len(nextBatch)
                    self.trainQueue.put(("trainBatch", trainId, nextBatchPrepared, iterationNumber, frameNumber / iterationSize))

                if time.monotonic() - lastPrint > 30:
                    lastPrint = time.monotonic()
                    progressPerc = (frameNumber / iterationSize) * 100.0
                    logMsg("Iteration %i: %.2f%% " % (iterationNumber, progressPerc))

            logMsg("Iteration %i completed" % (iterationNumber))

            self.publishWait = True
            self.trainQueue.put(("publish", iterationNumber))

            logMsg("Now waiting to publish the network before the next iteration begins!")
            while self.publishWait:
                time.sleep(0.1)

    def manage(self):
        logMsg("Starting StreamTraining will continue once first states show up on the server!")
        while len(self.serverStates) == 0:
            time.sleep(1)
        logMsg("The server has %i packages available to learn from" % len(self.serverStates))

        # wait for the windowBuffer to fill to the minimal size
        minSize = self.windowManager.getMinimumWindowSize()

        if minSize > len(self.windowBuffer):
            logMsg("Waiting for the window buffer to grow to the minimum size of", minSize)

        while minSize > len(self.windowBuffer):
            self.windowBuffer.append(blockGet(self.newQueue))

        random.shuffle(self.windowBuffer)

        self.manageLoop()
            


class StreamTrainingWorker2():
    def __init__(self, policy, windowManager, batchSize = 1024, deduplicate = False, deduplicationWeight = 0.5):
        hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for the trainer: --secret <server password>, --run <uuid> and --command <command server host>!")

        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.runId = sys.argv[sys.argv.index("--run")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]

        self.policy = policy
        self.windowManager = windowManager
        self.networks = NetworkApi()
        self.batchSize = batchSize
        self.deduplicate = deduplicate
        self.deduplicationWeight = deduplicationWeight

    def initManager(self):
        self.trainQueue = mp.Queue()
        self.trainEventsQueue = PlainQueue()

        self.proc = mp.Process(target=manageStreamWork, args=(self.trainQueue, self.trainEventsQueue, self.commandHost, self.secret, self.runId, self.windowManager, self.batchSize, self.policy.getExamplePrepareObject(), self.deduplicate, self.deduplicationWeight))
        self.proc.start()

        # helpful to debug, the sub process can hide some types of errors (like segfaults...)
        #manageStreamWork(self.trainQueue, self.trainEventsQueue, self.commandHost, self.secret, self.runId, self.windowManager, self.batchSize, self.policy.getExamplePrepareObject())

    def loopGpu(self):
        fitTimes = []
        getTimes = []
        logMsg("Begin gpu loop")
        while True:
            startLoop = time.monotonic_ns()
            fitTime = 0
            getStart = time.monotonic_ns()
            nextWork = self.trainQueue.get()
            getTime = time.monotonic_ns() - getStart

            if nextWork[0] == "trainBatch":
                startFit = time.monotonic_ns()
                fitResult = self.policy.fit(nextWork[2], nextWork[3], nextWork[4])
                endFit = time.monotonic_ns()
                fitTime = endFit - startFit

                if fitResult is None:
                    wls = []
                    mls = []
                else:
                    mls, wls = fitResult

                self.trainEventsQueue.put(("finish", nextWork[1], mls, wls))
            elif nextWork[0] == "publish":
                iteration = nextWork[1]
                self.trainEventsQueue.put(("network", iteration, self.policy.getUUID(), encodeToBson(self.policy.store())))

            allTime = (time.monotonic_ns() - startLoop)
            fitTimes.append(fitTime / allTime)
            getTimes.append(getTime / allTime)

            if nextWork[0] == "publish":
                logMsg("Spending %.2f%% of time fitting the network. Waiting for new data %.2f%% of the time" % (np.mean(fitTimes) * 100, np.mean(getTimes) * 100))
                fitTimes = []
                getTimes = []

            del nextWork

    def main(self):
        mp.set_start_method("spawn")
        self.initManager()
        
        # do the actual training by interacting with the queues
        
        self.networks.loadNewestNetwork(self.policy)

        try:
            self.loopGpu()
        finally:
            logMsg("Kill manager process!")
            self.proc.kill()
