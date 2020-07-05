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

        # UUID -> ID of the game batch in self.games
        self.pendingEvals = dict()
    
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

    def requestEvaluationForBatch(self, batchId):
        self.numPositionEvalsRequested += len(self.games[batchId])
        for game in self.games[batchId]:
            self.pendingGames.add(game)
        newPackageId = self.evalAccess.requestEvaluation(self.games[batchId])
        self.pendingEvals[newPackageId] = batchId
        return newPackageId

    def checkForRejectEvals(self, evalResults):
        acceptedEvals = 0
        for uuid in list(dict.keys(evalResults)):
            results, network, workerName = evalResults[uuid]
            if network != self.currentNetwork:
                gamesId = self.pendingEvals[uuid]
                del self.pendingEvals[uuid]
                del evalResults[uuid]

                newPackageId = self.requestEvaluationForBatch(gamesId)
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
        # walk over all the active batches that are not waiting for an evaluation and play them out, so they need an evaluation  
        pendingIdx = set(dict.values(self.pendingEvals))
        for batchIndex in range(len(self.games)):
            if not (batchIndex in pendingIdx):
                self.playOnBatch(batchIndex)

    def playOnBatch(self, bindex):
        # play a batch until all positions in it are not known in the cache. If a game ends, report the game and replace it with a new one.
        # once all positions in a batch need an evaluation, request it
        batch = self.games[bindex]

        blocked = False

        for gidx in range(len((batch))):
            agame = batch[gidx]

            while True:
                batch[gidx] = agame
                if agame in self.pendingGames:
                    # this batch cannot continue, until some other batch is finished...
                    blocked = True
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
                    # this index now needs an evaluation to continue!
                    break

        if not blocked:
            newPackageId = self.requestEvaluationForBatch(bindex)
            logMsg("Requested evaluation of batch %i with id %s" % (bindex, newPackageId))
        else:
            logMsg("Batch %i is blocked" % bindex)

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

    def addResultsToCache(self, batchId, evalResult):
        results, network, workerName = evalResult
        assert network == self.currentNetwork
        gameBatch = self.games[batchId]
        for idx, result in enumerate(results):
            gameState = gameBatch[idx]
            self.pendingGames.remove(gameState)
            self.cache[gameState] = result

    def receiveGameEvals(self):
        evalResults = dict()
        while len(evalResults) == 0:
            evalResults = self.evalAccess.pollEvaluationResults()
            self.checkForNewIteration(evalResults)
            self.checkForRejectEvals(evalResults)
        
        for uuid in evalResults:
            if uuid in self.pendingEvals:
                batchId = self.pendingEvals[uuid]
                del self.pendingEvals[uuid]
                self.addResultsToCache(batchId, evalResults[uuid])
                logMsg("Continue to play batch %i after receiving package %s" % (batchId, uuid))
                self.playOnBatch(batchId)
            else:
                logMsg("Received evaluation of unknown UUID!", uuid)

    def playBatch(self):
        self.playOpenGames()
        self.receiveGameEvals()

        # not meant to be used for frametime evaluations
        return 0, None, 0
        

            