import abc

from core.selfplay.SelfPlayWorker import GameReporter, PolicyUpdater
from utils.prints import logMsg

import time

reportedData = []
iterationReports = 0
iterationCounter = 1
needsFitting = False

class SingleProcessReporter(GameReporter, metaclass=abc.ABCMeta):
    
    def __init__(self, windowSize, reportsPerIteration):
        logMsg("Initialize SingleProcessReporter windowSize=%i, reportsPerIteration=%i" % (windowSize, reportsPerIteration))
        self.windowSize = windowSize
        self.reportsPerIteration = reportsPerIteration
        global reportedData
        global iterationCounter
        global needsFitting
        global iterationReports
        reportedData = []
        iterationReports = 0
        iterationCounter = 1
        needsFitting = False
        
    def reportGame(self, reports):
        global reportedData
        global iterationCounter
        global needsFitting
        global iterationReports

        for report in reports:
            reportedData.append(report)
            iterationReports += 1

        while len(reports) > self.windowSize:
            del reports[0]

        if iterationReports > self.reportsPerIteration:
            logMsg("Finished iteration %i with %i reports" % (iterationCounter, iterationReports))
            iterationReports = 0
            iterationCounter += 1
            needsFitting = True

class SingleProcessUpdater(PolicyUpdater, metaclass=abc.ABCMeta):

    def __init__(self, trainEpochs):
        self.trainEpochs = trainEpochs

    def update(self, policy):
        global reportedData
        global needsFitting

        if needsFitting:
            policy.fit(reportedData, self.trainEpochs)
            needsFitting = False