# cython: profile=False

from core.playing.SelfPlayWorker import SelfPlayWorker

import uuid

import abc

from utils.prints import logMsg

import time

import numpy as np
import datetime

import random

class SelfPlayMoveDecider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decideMove(self, gameState, policyDistribution, extraStats):
        """
        @return move to be played based on the gameState and the policy and extra stats found by the policy iterator.
        """

class LinearSelfPlayWorker(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initialState, policy, policyIterator, gameCount, moveDecider, gameReporter, policyUpdater, capPolicyP = None, capPolicyIterator = None, capMoveDecider = None):
        """
        To use playout cap randomization (see arXiv:1902.10565v3), configure:
        capPolicyP: what fraction of the games should be capped (this is 1-p in the paper, I find that more easy to understand)
        capPolicyIterator: policy iterator to use for capped play steps. E.g. MCTS with a lower expansions setting. Possibly also less cpuct, less fpu, no rootNoise, lower drawValue to reduce exploration and maximize strength in these moves.
        capMoveDecider: moveDecider to use on capped moves. Can be omitted, then the normal moveDecider will be used
        """
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

        self.capPolicyP = capPolicyP
        self.capPolicyIterator = capPolicyIterator
        self.capMoveDecider = capMoveDecider

        assert (self.capPolicyP is None and self.capPolicyIterator is None) or (self.capPolicyP is not None and self.capPolicyIterator is not None), "capPolicyP and capPolicyIterator both need to be set or not set"

        if self.capPolicyP is not None:
            logMsg("Playout Cap Randomization is enabled, %i%% of moves will be played with a faster policy iterator and not recorded as training material to playout more games at little cost!" % (self.capPolicyP * 100))

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
        self.initSelfplay(None)

        while True:
            self.playBatch()

    def initSelfplay(self, runId):
        self.policy = self.policyUpdater.update(self.policy)
        assert not (self.policy is None), "the policy updater returned a None policy!"

    def playBatch(self):
        """
        For the frametime measurement, this has to measure how long it takes to generate a single frame of training data and return that.
        So if multiple moves are played per frame of training data, that has all to be measured.
        """
        moveTimeNanos = 0

        while True:
            playRealMove = self.capPolicyP is None or random.random() >= self.capPolicyP

            policyIterator = self.policyIterator
            moveDecider = self.moveDecider
            if not playRealMove:
                policyIterator = self.capPolicyIterator
                if self.capMoveDecider is not None:
                    moveDecider = self.capMoveDecider

            # generate iterated policy to be played on
            iteratationStart = time.monotonic_ns()
            iteratedPolicy = policyIterator.iteratePolicy(self.policy, self.open)
            moveTimeNanos += time.monotonic_ns() - iteratationStart

            if playRealMove:
                # remember the iterated policy for each game to be later processed into training frames.
                self.addTrackingData(iteratedPolicy)

                # potentially update the policy with a new policy from somewhere (e.g. a training server)
                self.initSelfplay(None)

            # play the moves using the iterated policy.
            playStart = time.monotonic_ns()
            movesToPlay = list(map(lambda x: moveDecider.decideMove(x[0], x[1][0], x[1][1]), zip(self.open, iteratedPolicy)))
            self.open = list(map(lambda x: x[0].playMove(x[1]), zip(self.open, movesToPlay)))
            moveTimeNanos += time.monotonic_ns() - playStart

            # go over finished games and replace them with new ones, reporting their frames to the server
            self.finalizeGames()

            if playRealMove:
                break

        # sum up the time used to iterate the policy and play the move
        usTime = int((moveTimeNanos / float(len(self.open))) / 1000)
        self.microsPerMovePlayedHistory.append(usTime)
        self.handleSpeedStats()

        return (usTime / 1000.0), None, len(self.open)

    def addTrackingData(self, iteratedPolicy):
        for idx, game in enumerate(self.open):
            if self.tracking[idx] == None:
                self.tracking[idx] = []
            
            self.tracking[idx].append([game, iteratedPolicy[idx]])

    def handleReportsFor(self, idx):
        reports = []

        trackList = self.tracking[idx]

        # this can happen with playout cap randomization, if an entire game is played with the cheap policy.
        # that tends to happen espacially with an untrained network that plays very bad moves.
        if trackList is None:
            return

        assert self.open[idx].hasEnded()
        result = self.open[idx].getWinnerNumber()
        policyUUID = self.policy.getUUID()

        prevStateUUID = None

        for ti in range(len(trackList)):
            state, iPolicy = trackList[ti]
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
            record["creation"] = datetime.datetime.utcnow().timestamp()
            record["final"] = False

            # if the game is over after the current move is played, encode that as a uniform distribution over all moves
            reply = np.ones_like(record["policyIterated"])
            reply /= reply.shape[0]
            if (ti + 1) < len(trackList):
                # if however there is another move played, track the distribution for that move
                reply = trackList[ti+1][1][0]
            record["reply"] = reply

            reports.append(record)

        self.tracking[idx] = None

        if len(reports) > 0:
            reports[len(reports) - 1]["final"] = True
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



