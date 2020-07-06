import abc
import uuid
import datetime
from utils.prints import logMsg
from core.playing.SelfPlayWorker import SelfPlayWorker

from utils.req import requestJson, postJson, requestBytes, postBytes
from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

import time

import sys

import numpy as np

class CachedLinearSelfPlay(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initialState, batchSize, numActiveBatches, gameReporter, evalAccess, moveDecider):

        self.commandHost = sys.argv[sys.argv.index("--command")+1]
        self.run = sys.argv[sys.argv.index("--run")+1]
        self.secret = sys.argv[sys.argv.index("--secret")+1]

        self.initialState = initialState
        self.batchSize = batchSize
        self.numActiveBatches = numActiveBatches
        self.gameReporter = gameReporter
        self.evalAccess = evalAccess
        self.moveDecider = moveDecider

        self.games = dict()
        self.histories = []
        for i in range(numActiveBatches):
            self.games[i] = [self.initialState] * batchSize
            self.histories.append([])
            for _ in range(batchSize):
                self.histories[i].append([])

        # UUID -> list of game states that will be evaluated
        self.pendingEvals = dict()
        # set of game states to be evaluated at the next opportunity
        self.requestEvals = set()

        # set of positions that are already pending for evaluation
        self.pendingGames = set()

        # state -> MCTS result for that state, if known
        self.cache = dict()

        self.seenNetworks = set()
        self.currentNetwork = None
        self.emptyNetworkID = str(uuid.uuid4())

        self.numPositionEvalsRequested = 0

    def reportEvaluationNumber(self):
        drep = dict()
        drep["evals"] = self.numPositionEvalsRequested
        drep["iteration"] = len(self.seenNetworks) - 1
        postJson(self.commandHost + "/api/evalscnt/" + self.run, self.secret, drep)
        logMsg("Iteration %i used %i evaluation requests!" % (drep["evals"], drep["iteration"]))
        self.numPositionEvalsRequested = 0

    def checkForNewIteration(self, evalResults):
        for uuid in evalResults:
            results, networkSeen, workerName = evalResults[uuid]
            if networkSeen is not None and not (networkSeen in self.seenNetworks):
                self.seenNetworks.add(networkSeen)
                self.currentNetwork = networkSeen
                self.cache = dict()
                logMsg("A new iteration has begun, cleared the cache!")

                self.reportEvaluationNumber()

    def checkForRejectEvals(self, evalResults):
        acceptedEvals = 0
        for uuid in list(dict.keys(evalResults)):
            results, network, workerName = evalResults[uuid]
            if network != self.currentNetwork:
                gamePackage = self.pendingEvals[uuid]
                del self.pendingEvals[uuid]
                del evalResults[uuid]

                newPackageId = self.evalAccess.requestEvaluation(gamePackage)
                self.pendingEvals[newPackageId] = gamePackage
                logMsg("!!!!!!!!!! Got network %s but expected %s on package %s from worker %s. Rewriting to package %s" % (network, self.currentNetwork, uuid, workerName, newPackageId))
            else:
                acceptedEvals += 1

        if acceptedEvals > 0:
            logMsg("Accepted evaluation of %i batches!" % acceptedEvals)

    def main(self):
        while True:
            self.playBatch()

    def initSelfplay(self, runId):
        pass

    def playOpenGames(self):
        for batchIndex in range(len(self.games)):
            self.playOnBatch(batchIndex)

    def requestEvaluations(self):
        while (len(self.pendingEvals) == 0 and len(self.requestEvals) > 0) or len(self.requestEvals) >= self.batchSize:
            nextBatch = list(self.requestEvals)[:self.batchSize]
            newPackageId = self.evalAccess.requestEvaluation(nextBatch)
            self.pendingEvals[newPackageId] = nextBatch
            for game in nextBatch:
                self.pendingGames.add(game)
                self.requestEvals.remove(game)

    def playOnBatch(self, bindex):
        # play a batch until all positions in it are not known in the cache. If a game ends, report the game and replace it with a new one.
        # once all positions in a batch need an evaluation, request it
        batch = self.games[bindex]

        for gidx in range(len((batch))):
            agame = batch[gidx]

            while True:
                batch[gidx] = agame
                if agame in self.pendingGames:
                    # cannot continue this game at this time, it is already pending for an evaluation.
                    break
                elif agame.hasEnded():
                    self.finalizeGame(bindex, gidx)
                    agame = self.initialState
                elif agame in self.cache:
                    iteratedPolicy = self.cache[agame]
                    self.addTrackingData(agame, iteratedPolicy, bindex, gidx)
                    moveToPlay = self.moveDecider.decideMove(agame, iteratedPolicy[0], iteratedPolicy[1])
                    agame = agame.playMove(moveToPlay)
                else:
                    self.requestEvals.add(agame)
                    break

        self.requestEvaluations()

    def addTrackingData(self, game, iteratedPolicy, batchIndex, gameIndex):
        self.histories[batchIndex][gameIndex].append([game, iteratedPolicy])

    def finalizeGame(self, batchIndex, gameIndex):
        history = self.histories[batchIndex][gameIndex]
        self.histories[batchIndex][gameIndex] = []

        reports = []

        assert self.games[batchIndex][gameIndex].hasEnded()
        result = self.games[batchIndex][gameIndex].getWinnerNumber()
        policyUUID = self.currentNetwork
        if policyUUID is None:
            policyUUID = self.emptyNetworkID

        for ti in range(len(history)):
            state, iPolicy = history[ti]
            assert not state.hasEnded()
            record = dict()
            record["gameCtor"] = state.getGameConstructorName()
            record["gameParams"] = state.getGameConstructorParams()
            record["knownResults"] = [result]
            record["generics"] = dict(iPolicy[1])
            record["policyIterated"] = iPolicy[0]
            record["uuid"] = str(uuid.uuid4())
            record["policyUUID"] = policyUUID
            record["state"] = state.store()
            record["gamename"] = state.getGameName()
            record["creation"] = datetime.datetime.utcnow().timestamp()
            record["final"] = False

            # if the game is over after the current move is played, encode that as a uniform distribution over all moves
            reply = np.ones_like(record["policyIterated"])
            reply /= reply.shape[0]
            if (ti + 1) < len(history):
                # if however there is another move played, track the distribution for that move
                reply = history[ti+1][1][0]
            record["reply"] = reply

            reports.append(record)

        if len(reports) > 0:
            reports[-1]["final"] = True
            self.gameReporter.reportGame(reports)

    def addResultsToCache(self, games, evalResult):
        results, network, workerName = evalResult
        assert network == self.currentNetwork
        for idx, result in enumerate(results):
            gameState = games[idx]
            self.pendingGames.discard(gameState)
            self.cache[gameState] = result

    def receiveGameEvals(self):
        evalResults = dict()
        while len(evalResults) == 0:
            evalResults = self.evalAccess.pollEvaluationResults()
            self.checkForNewIteration(evalResults)
            self.checkForRejectEvals(evalResults)
        
        for uuid in evalResults:
            if uuid in self.pendingEvals:
                evalGames = self.pendingEvals[uuid]
                del self.pendingEvals[uuid]
                self.addResultsToCache(evalGames, evalResults[uuid])
            else:
                logMsg("Received evaluation of unknown UUID!", uuid)

    def playBatch(self):
        self.playOpenGames()
        self.receiveGameEvals()

        # not meant to be used for frametime evaluations
        #frametime is evaluated instead on the evluation worker.
        return 0, None, 0
        

            