from core.training.StatesDownloader import StatesDownloader
from core.training.NetworkApi import NetworkApi

import time

from utils.prints import logMsg

from utils.bsonHelp.bsonHelp import decodeFromBson

import abc

import random

import numpy as np

class WindowSizeManager(metaclass=abc.ABCMeta):
    """
    Responsible for deciding the size of the training window for the current iteration
    """

    @abc.abstractmethod
    def getWindowSize(self, currentIteration):
        """
        @return a number that describes how many frames should be in the training window for the given iteration
        at most, if there is more data available the StreamTrainingWorker will throw away all states and
        not learn from them anymore
        """

    @abc.abstractmethod
    def getIterationSize(self, currentIteration):
        """
        @return how many new states should be used for training until a new network generation is published
        """

    @abc.abstractmethod
    def getMinimumWindowSize(self):
        """
        @return how many states must there be in the trainingWindow before the next iteration
        is planned and the network training begins.
        """

class ConstantWindowSizeManager(WindowSizeManager, metaclass=abc.ABCMeta):
    def __init__(self, size, minimumSize, iterationSize):
        self.size = size
        self.minimumSize = minimumSize
        self.iterationSize = iterationSize

    def getWindowSize(self, currentIteration):
        return self.size

    def getIterationSize(self, currentIteration):
        return self.iterationSize

    def getMinimumWindowSize(self):
        return self.minimumSize


class StreamTrainingWorker():
    def __init__(self, policy, windowManager, batchSize):
        """
        Requires some configuration parameters to be present in the arguments to python
        --command <command server host>
        --secret <server api password>
        --run <run-uuid>
        """

        self.policy = policy;

        self.networks = NetworkApi()
        self.datadir = "streamTrainingData/" + self.networks.runId
        self.downloader = StatesDownloader(self.datadir)
        self.batchSize = batchSize

        # pool of game state records that can be picked from for the next mini batch
        # gets filled by newly generated states and drained by the training of the network
        # at a factor that results in it being empty once enough new game states have been generated 
        # for a full network iteration
        self.windowBuffer = []

        # waiting room for game states that have already been learned from, but might be
        # reused in the next iteration. Gets drained back into the windowBuffer,
        # when the current network iteration is finished
        # states will be prioritized by age, if the windowBuffer has a max size,
        # older states will not be put back and instead dropped entirely.
        self.waitBuffer = []
        
        self.windowManager = windowManager

        self.openPacks = set()

    def waitForNewStates(self, atLeastN):
        """
        Blocks and waits until atLeastN new states are available (the statesdownloader downloads them in parallel) to train from.
        Returns a list of all newly available states to learn from.
        """

        newStatesLst = []

        while len(newStatesLst) < atLeastN:
            unknowns = filter(lambda x: not (x["id"] in self.openPacks), self.downloader.history.copy())
            for unknown in unknowns:
                self.openPacks.add(unknown["id"])
                for state in self.downloader.openPackage(unknown["id"]):
                    newStatesLst.append(state)
            time.sleep(0.2)

        return newStatesLst

    def addNewFrames(self, nextFrames):
        self.windowBuffer += nextFrames
        return len(nextFrames)

    def pickFramesForTraining(self, num):
        random.shuffle(self.windowBuffer)
        picked = self.windowBuffer[:num]
        self.windowBuffer = self.windowBuffer[num:]
        self.waitBuffer += picked
        return picked

    def main(self):
        self.downloader.start()

        self.networks.loadNewestNetwork(self.policy)

        # wait for the windowBuffer to fill to the minimal size
        minSize = self.windowManager.getMinimumWindowSize()
        logMsg("Waiting for the window buffer to grow to the minimum size of", minSize)
        nextFrames = self.waitForNewStates(minSize)
        self.addNewFrames(nextFrames)

        logMsg("The window buffer has now %i states in it, continuing..." % len(self.windowBuffer))

        while True:
            iterationStart = time.monotonic()

            iterationNumber = len(self.networks.getNetworkList()) + 1
            iterationSize = self.windowManager.getIterationSize(iterationNumber)
            oldDataSize = len(self.windowBuffer)
            sumStatesCount = iterationSize + oldDataSize
            framesPerNewFrame = sumStatesCount / iterationSize
            pendingTrainingCount = 0

            logMsg("Beginning a new network iteration, we're in iteration %i" % iterationNumber)
            logMsg("We have %i old states, want to fit on %i states, this requires %.2f examples learnt per new state" % (oldDataSize, sumStatesCount, framesPerNewFrame))

            newFramesCount = 0

            gmls = []
            gwls = []

            proc = 0

            while newFramesCount < iterationSize:
                addedFrames = self.addNewFrames(self.waitForNewStates(self.batchSize))

                pendingTrainingCount += addedFrames * framesPerNewFrame
                batches = pendingTrainingCount // self.batchSize
                trainFrameCount = int(batches * self.batchSize)
                pendingTrainingCount -= trainFrameCount

                trainingFrames = self.pickFramesForTraining(trainFrameCount)

                fitResult = self.policy.fit(trainingFrames, 1)

                if fitResult is not None:
                    mls, wls = fitResult
                    gmls += mls
                    gwls += wls

                newFramesCount += addedFrames

                procCur = int((newFramesCount / iterationSize) * 100.0) // 10
                if procCur > proc:
                    proc = procCur
                    logMsg("Iteration %i%% completed: learnt from %i frames. Window buffer currently contains %i frames" %(proc * 10, newFramesCount, len(self.windowBuffer)))


            logMsg("Iteration completed, new frames processed: %i." % newFramesCount)
            logMsg("Iteration network loss: %.2f on moves, %.2f on outcome" % (np.mean(gmls), np.mean(gwls)))
            self.networks.uploadNetwork(self.policy)

            nextWindowSize = self.windowManager.getWindowSize(iterationNumber)

            cpyc = 0
            self.waitBuffer.sort(key = lambda x: x["creation"], reverse=True)
            for frame in self.waitBuffer:
                if len(self.windowBuffer) < nextWindowSize:
                    cpyc += 1
                    self.windowBuffer.append(frame)
                else:
                    break
            dropped = len(self.waitBuffer) - cpyc
            self.waitBuffer = []
            
            iterationEnd = time.monotonic()
            logMsg("Moved %i states back into the window from the waiting buffer. Dropped %i old ones. Iteration finished, iteration time: %.2fs" % (cpyc, dropped, iterationEnd - iterationStart))

