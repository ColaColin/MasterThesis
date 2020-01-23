# cython: profile=False

from core.playing.SelfPlayWorker import SelfPlayWorker

import uuid

import abc

from utils.prints import logMsg

import time

import numpy as np

class SelfPlayMoveDecider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decideMove(self, gameState, policyDistribution, extraStats):
        """
        @return move to be played based on the gameState and the policy and extra stats found by the policy iterator.
        """

class LinearSelfPlayWorker(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initialState, policy, policyIterator, gameCount, moveDecider, gameReporter, policyUpdater):
        logMsg("Creating LinearSelfPlayWorker gameCount=%i" % gameCount)
        self.initialState = initialState
        self.policy = policy
        self.policyIterator = policyIterator
        self.open = [initialState] * gameCount
        assert not initialState.hasEnded()
        self.tracking = [None] * gameCount;
        self.moveDecider = moveDecider
        self.microsPerMovePlayedHistory = []
        self.gameReporter = gameReporter
        self.policyUpdater = policyUpdater
        self.lastSpeedStatsPrint = time.monotonic()

    def handleSpeedStats(self):
        if time.monotonic() - self.lastSpeedStatsPrint > 300:
            self.lastSpeedStatsPrint = time.monotonic()
            count = len(self.microsPerMovePlayedHistory)
            mean = np.mean(self.microsPerMovePlayedHistory)
            median = np.median(self.microsPerMovePlayedHistory)
            std = int(np.std(self.microsPerMovePlayedHistory))

            logMsg("Last %i loops move times: %i us median, %i us mean,  +/- %i us" % (count, median, mean, std))

            self.microsPerMovePlayedHistory = []

    def main(self):
        self.selfplay()

    def selfplay(self):
        self.policy = self.policyUpdater.update(self.policy)
        while True:
            moveTimeNanos = 0

            iteratationStart = time.monotonic_ns()
            iteratedPolicy = self.policyIterator.iteratePolicy(self.policy, self.open)
            moveTimeNanos += time.monotonic_ns() - iteratationStart

            self.addTrackingData(iteratedPolicy)
            self.policy = self.policyUpdater.update(self.policy)
            assert not (self.policy is None), "the policy updater returned a None policy!"

            playStart = time.monotonic_ns()
            movesToPlay = list(map(lambda x: self.moveDecider.decideMove(x[0], x[1][0], x[1][1]), zip(self.open, iteratedPolicy)))
            self.open = list(map(lambda x: x[0].playMove(x[1]), zip(self.open, movesToPlay)))
            moveTimeNanos += time.monotonic_ns() - playStart

            self.microsPerMovePlayedHistory.append(int((moveTimeNanos / float(len(self.open))) / 1000))
            self.handleSpeedStats()

            self.finalizeGames()

    def addTrackingData(self, iteratedPolicy):
        for idx, game in enumerate(self.open):
            if self.tracking[idx] == None:
                self.tracking[idx] = []
            
            self.tracking[idx].append([game, iteratedPolicy[idx]])


    def handleReportsFor(self, idx):
        reports = []

        trackList = self.tracking[idx]
        assert self.open[idx].hasEnded()
        result = self.open[idx].getWinnerNumber()
        policyUUID = self.policy.getUUID()

        prevStateUUID = None
        for state, iPolicy in trackList:
            # do not learn from terminal states, there is no move that can be made on them
            assert not state.hasEnded()
            record = dict()
            record["gameCtor"] = state.getGameConstructorName()
            record["gameParams"] = state.getGameConstructorParams()
            record["knownResults"] = [result]
            record["generics"] = dict(iPolicy[1])
            record["policyIterated"] = iPolicy[0]
            record["uuid"] = str(uuid.uuid4())
            record["parent"] = prevStateUUID
            prevStateUUID = record["uuid"]
            record["policyUUID"] = policyUUID
            record["state"] = state.store()
            record["gamename"] = state.getGameName()
            reports.append(record)
            
        self.tracking[idx] = None

        if len(reports) > 0:
            self.gameReporter.reportGame(reports)

    def finalizeGames(self):
        """
        replace games that have been completed with new games, if desired by the impl.
        After this call all games in self.open should not be in a terminal state.
        The default impl just filters games out and replaces them with new games.
        """
        cdef int idx

        newList = []
        for idx, o in enumerate(self.open):
            if o.hasEnded():
                self.handleReportsFor(idx)
                newList.append(self.initialState)
            else:
                newList.append(o)

        self.open = newList



