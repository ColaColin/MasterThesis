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

import random

class CachedNonLinearSelfPlay(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initialState, batchSize, numActiveBatches, gameReporter, evalAccess, moveDecider, pickMode="winp", initPoints=8):

        self.commandHost = sys.argv[sys.argv.index("--command")+1]
        self.run = sys.argv[sys.argv.index("--run")+1]
        self.secret = sys.argv[sys.argv.index("--secret")+1]

        self.initialState = initialState
        self.batchSize = batchSize
        self.numActiveBatches = numActiveBatches
        self.gameReporter = gameReporter
        self.evalAccess = evalAccess
        self.moveDecider = moveDecider

        # a game is described by a list: [state, gameHistory]
        # that way each of these games has the history for reporting and there is no need for separate history tracking...
        self.activeGames = []

        for _ in range(initPoints):
            self.activeGames.append([self.initialState, []])

        # UUID -> list of game states that will be evaluated
        self.pendingEvals = dict()
        # set of game states to be evaluated at the next opportunity
        self.requestEvals = set()

        # set of positions that are already pending for evaluation
        self.pendingGames = set()

        # state -> MCTS result for that state, if known
        self.cache = dict()

        self.pickMode = pickMode

        # a set of tuples (state, move) for every move played in the current iteration
        self.playedMoves = set()

        self.seenNetworks = set()
        self.currentNetwork = None
        self.emptyNetworkID = str(uuid.uuid4())

        self.numPositionEvalsRequested = 0

        self.retryTurns = []

        self.foundRetry = 0
        self.foundNoRetry = 0

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
            if not (networkSeen is None) and not (networkSeen in self.seenNetworks):
                self.seenNetworks.add(networkSeen)
                logMsg("seen nets is after", self.seenNetworks)
                self.currentNetwork = networkSeen
                self.cache = dict()
                self.playedMoves = set()
                logMsg("=======================================================")
                logMsg("=======================================================")
                logMsg("A new iteration has begun, cleared the cache! Successful retries: %.2f%%" % (100 * (self.foundRetry / (1 + self.foundRetry + self.foundNoRetry))))
                logMsg("Retry at turn %.2f +/- %.2f" % (np.mean(self.retryTurns), np.std(self.retryTurns)))
                logMsg("Network now is %s" % self.currentNetwork)
                logMsg("=======================================================")
                logMsg("=======================================================")

                self.foundNoRetry = 0
                self.foundRetry = 0
                self.retryTurns = []

                self.reportEvaluationNumber()

    def checkForRejectEvals(self, evalResults):
        acceptedEvals = 0
        for uuid in list(dict.keys(evalResults)):
            results, network, workerName = evalResults[uuid]
            if network != self.currentNetwork:
                gamePackage = self.pendingEvals[uuid]
                del self.pendingEvals[uuid]
                del evalResults[uuid]
                
                self.numPositionEvalsRequested += len(gamePackage)
                newPackageId = self.evalAccess.requestEvaluation(gamePackage)
                self.pendingEvals[newPackageId] = gamePackage
                logMsg("!!!!!!!!!! Got network %s but expected %s on package %s from worker %s. Rewriting to package %s" % (network, self.currentNetwork, uuid, workerName, newPackageId))
            else:
                acceptedEvals += 1

        if acceptedEvals > 0 and False:
            logMsg("Accepted evaluation of %i batches!" % acceptedEvals)

    def main(self):
        while True:
            self.playBatch()

    def initSelfplay(self, runId):
        pass

    def pickRetryPoints(self, history, finalState):
        # here do the "go back and fix your errors thing"

        if finalState.getWinnerNumber() == 0:
            # draw
            loser = random.randint(1, finalState.getPlayerCount())
        if finalState.getWinnerNumber() == 1:
            loser = 2
        else:
            loser = 1

        needResults = 4
        if len(self.activeGames) >= self.batchSize * self.numActiveBatches:
            needResults = 1

        def collectBestMovesNeverTried(history, forPlayer, playedMovesSet):
            bestMoves = []
            for hidx, (state, iPolicy0) in enumerate(history):
                if state in self.cache:
                    iPolicy = self.cache[state]
                else:
                    iPolicy = iPolicy0
                if state.getPlayerOnTurnNumber() == forPlayer:
                    movePs = iPolicy[0]
                    for m in state.getLegalMoves():
                        if not ((state, m) in playedMovesSet):
                            bestMoves.append([movePs[m], m, hidx])

            bestMoves.sort(key=lambda x: x[0], reverse=True)
            return bestMoves

        # retry points are 2 element lists, of [state, history]
        results = []

        if self.pickMode == "winp":
            # greatest win probability loss, then best move never tried in that position, if there is none, try the next best position...

            positionsByValueLoss = []

            for hi in range(len(history)):
                istate, ipolicy = history[hi]
                if istate in self.cache:
                    ipolicy = self.cache[istate]
                ivalue = ipolicy[1]["net_values"][loser] + 0.5 * ipolicy[1]["net_values"][0]
                if istate.getPlayerOnTurnNumber() == loser:
                    nextMyState = hi + 1
                    while nextMyState < len(history):
                        if history[nextMyState][0].getPlayerOnTurnNumber() == loser:
                            break
                        else:
                            nextMyState += 1

                    if nextMyState < len(history):
                        nextstate, nextpolicy = history[nextMyState]
                        if nextstate in self.cache:
                            nextpolicy = self.cache[nextstate]
                        nextvalue = nextpolicy[1]["net_values"][loser] + 0.5 * nextpolicy[1]["net_values"][0]

                        #print("Loss found between positions", "\n", str(istate), "\n", str(nextstate), "\n=>", ivalue, nextvalue, ivalue - nextvalue)

                        valueLoss = ivalue - nextvalue

                        positionsByValueLoss.append([valueLoss, hi])

            positionsByValueLoss.sort(key=lambda x: x[0], reverse=True)

            #print("biggest losses: ", positionsByValueLoss[:5])

            stop = False

            for loss, hindex in positionsByValueLoss:
                if stop:
                    break
                bestMoves = collectBestMovesNeverTried([history[hindex]], loser, self.playedMoves)
                #print("legal moves in position on index", hindex, bestMoves)
                pickedMoves = bestMoves[:needResults]
                for pm in pickedMoves:
                    if len(results) >= needResults:
                        stop = True
                        break

                    move = pm[1]
                    
                    curState = history[hindex][0]
                    assert move in curState.getLegalMoves()
                    nextPosition = curState.playMove(move)
                    self.playedMoves.add((curState, move))

                    result = [nextPosition, history[:hindex+1]]
                    results.append(result)

                if len(results) >= needResults:
                    stop = True

        elif self.pickMode == "bestm":
            # best move never tried

            bestMoves = collectBestMovesNeverTried(history, loser, self.playedMoves)

            pickedMoves = bestMoves[:needResults]

            for pimove in pickedMoves:
                move = pimove[1]
                hindex = pimove[2]

                curState = history[hindex][0]
                assert move in curState.getLegalMoves()
                nextPosition = curState.playMove(move)
                self.playedMoves.add((curState, move))

                result = [nextPosition, history[:hindex+1]]
                results.append(result)

        else:
            assert False, "unknown pickmode: " + self.pickMode

        # print("Picked retry points:\n")
        # for result in results:
        #     print(str(result[0]))
        # print("----")

        return results

    def playOpenGames(self):
        # play a batch until all positions in it are not known in the cache. If a game ends, report the game and replace it with a new one.
        # once all positions in a batch need an evaluation, request it

        addedGames = False

        for gidx in range(len(self.activeGames)):
            agame, ahistory = self.activeGames[gidx]

            while True:
                self.activeGames[gidx] = [agame, ahistory]
                if agame in self.pendingGames:
                    # cannot continue this game at this time, it is already pending for an evaluation.
                    break
                elif agame.hasEnded():
                    self.finalizeGame(agame,ahistory)

                    if self.initialState in self.cache or self.initialState in self.pendingGames:
                        retryPoints = self.pickRetryPoints(ahistory, agame)
                        self.retryTurns += list(map(lambda x: x[0].getTurn(), retryPoints))
                        if len(retryPoints) > 0:
                            self.foundRetry += 1
                            agame, ahistory = retryPoints[0]
                            for anotherPoint in retryPoints[1:]:
                                self.activeGames.append(anotherPoint)
                                addedGames = True

                        else:
                            self.foundNoRetry += 1
                            logMsg("No retry points were generated!")
                            agame = self.initialState
                            ahistory = []
                    else:
                        logMsg("Initial state not evaluated in current iteration, starting a new game from the start!")
                        agame = self.initialState
                        ahistory = []

                elif agame in self.cache:
                    iteratedPolicy = self.cache[agame]
                    ahistory.append([agame, iteratedPolicy])
                    moveToPlay = self.moveDecider.decideMove(agame, iteratedPolicy[0], iteratedPolicy[1])
                    self.playedMoves.add((agame, moveToPlay))
                    agame = agame.playMove(moveToPlay)
                else:
                    self.requestEvals.add(agame)
                    break

        if addedGames:
            logMsg("There are now %i active games!" % len(self.activeGames))

        self.requestEvaluations()

    def requestEvaluations(self):
        while (len(self.pendingEvals) == 0 and len(self.requestEvals) > 0) or len(self.requestEvals) >= self.batchSize:
            nextBatch = list(self.requestEvals)[:self.batchSize]
            self.numPositionEvalsRequested += len(nextBatch)
            newPackageId = self.evalAccess.requestEvaluation(nextBatch)
            self.pendingEvals[newPackageId] = nextBatch
            for game in nextBatch:
                self.pendingGames.add(game)
                self.requestEvals.remove(game)

    def finalizeGame(self, game, history):
        reports = []

        assert game.hasEnded()
        result = game.getWinnerNumber()
        policyUUID = self.currentNetwork
        if policyUUID is None:
            policyUUID = self.emptyNetworkID

        for ti in range(len(history)):
            state, iPolicy = history[ti]
            if state in self.cache:
                iPolicy = self.cache[state]
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
        

            