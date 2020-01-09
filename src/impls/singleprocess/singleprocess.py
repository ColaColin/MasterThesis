import abc

from core.selfplay.SelfPlayWorker import GameReporter, PolicyUpdater
from utils.prints import logMsg

import numpy as np

import time
import pickle
import os

reportedData = []
iterationReports = 0
iterationCounter = 1
needsFitting = False

# TODO
# 3. Make an integrated test, that learns to play MNK(3,3,3) and checks if it plays correctly at some point?

class SingleProcessReporter(GameReporter, metaclass=abc.ABCMeta):
    
    def __init__(self, windowSize, reportsPerIteration, state):
        logMsg("Initialize SingleProcessReporter windowSize=%i, reportsPerIteration=%i,state=%s" % (windowSize, reportsPerIteration, state))
        self.windowSize = windowSize
        self.reportsPerIteration = reportsPerIteration
        self.state = state
        self.statePath = os.path.join(self.state, "state.pickle")
        self.lastStore = 0
        global reportedData
        global iterationCounter
        global needsFitting
        global iterationReports
        reportedData = []
        iterationReports = 0
        iterationCounter = 1
        needsFitting = False

        self.loadState()

    def storeState(self):
        global reportedData
        global iterationCounter
        global needsFitting
        global iterationReports

        startStore = time.time()

        package = dict()
        package["iterationReports"] = iterationReports
        package["iterationCounter"] = iterationCounter
        package["reportedData"] = reportedData

        with open(self.statePath, "wb") as f:
            pickle.dump(package, f)

        endStore = time.time()
        diffTime = endStore - startStore

        print("Storing reports took: %f" % diffTime)
        
    def loadState(self):
        global reportedData
        global iterationCounter
        global needsFitting
        global iterationReports

        if os.path.exists(self.statePath):
            with open(self.statePath, "rb") as f:
                package = pickle.load(f)
                iterationReports = package["iterationReports"]
                iterationCounter = package["iterationCounter"]
                needsFitting = False
                reportedData = package["reportedData"]

                print("Loaded previous progress of %i iterations" % iterationCounter)

    def reportGame(self, reports):
        global reportedData
        global iterationCounter
        global needsFitting
        global iterationReports

        for report in reports:
            reportedData.append(report)
            iterationReports += 1

        while len(reportedData) > self.windowSize:
            del reportedData[0]

        if iterationReports > self.reportsPerIteration:
            logMsg("Finished iteration %i with %i reports" % (iterationCounter, iterationReports))
            iterationReports = 0
            iterationCounter += 1
            needsFitting = True
            self.storeState()
           

class SingleProcessUpdater(PolicyUpdater, metaclass=abc.ABCMeta):

    def __init__(self, trainEpochs, state):
        logMsg("Initialized SingleProcessUpdater trainEpochs=%i, state=%s" % (trainEpochs, state))
        self.trainEpochs = trainEpochs
        self.state = state
        self.statePath = os.path.join(self.state, "policy.npy")
        self.loadedPolicyBytes = None
        self.loadState()

    def storeState(self, policy):
        np.save(self.statePath, policy.store(), allow_pickle=False)

    def loadState(self):
        if os.path.exists(self.statePath):
            self.loadedPolicyBytes = np.load(self.statePath)

    def update(self, policy):
        global reportedData
        global needsFitting

        if not (self.loadedPolicyBytes is None):
            policy.load(self.loadedPolicyBytes)
            self.loadedPolicyBytes = None
            print("Loaded stored policy with UUID %s!" % policy.getUUID())

        if needsFitting:
            policy.fit(reportedData, self.trainEpochs)
            needsFitting = False
            self.storeState(policy)

        return policy