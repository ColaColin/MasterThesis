import abc
import uuid
from utils.prints import logMsg
from core.playing.SelfPlayWorker import SelfPlayWorker

from utils.misc import hConcatStrings

from utils.req import requestJson, postJson

import time

import datetime
import numpy as np

import random

import sys

import threading

import math

# selfplay with "players" (i.e. mcts hyperparameter sets) that compete a in league against each other
# will not support playout caps, maybe they could be added in later, but for now they do not seem to help easily anyway.

class PlayerAccess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getNextMatch(self, runId):
        """
        @return two players to play a game against each other in the form of a tuple of tuples:
        (("player1-ID", {player1}), ("player2-ID", {player2}))
        """

    @abc.abstractmethod
    def reportResult(self, p1Id, p2Id, winnerId, policyUUID, runId):
        """
        give two player ids, and either player1 id, player2id or None for a draw.    
        """

class PlayerThinkDecider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def wantsToThinkBatch(self, piterators, remainings, currentIterations):
        """
        return a list of boolean, that contain True iff the iterator on that index wants to keep iterating
        """

class FixedThinkDecider(PlayerThinkDecider, metaclass=abc.ABCMeta):
    """
    think for a fixed number of iterations
    """
    def __init__(self, expansions):
        self.expansions = expansions

    def wantsToThinkBatch(self, piterators, remainings, currentIterations):
        return [self.expansions > ci for ci in currentIterations]

class LearntThinkDecider(PlayerThinkDecider, metaclass=abc.ABCMeta):
    """
    learn a few parameters that decide about thinking time
    """

    def __init__(self):
        self.lastIterResults = dict()
        self.lastIterCount = dict()
        self.kldgainsSeen = 1
        self.kldgainSum = 0.0001
        self.maxRemSeen = 0

    def shouldContinueThinking(self, params, stateValue, turn, avgExpansionsPlayed, entropyCurrent, entropyNetwork, kldgain):
        stateW = params["stateW"]
        turnW = params["turnW"]
        expW = params["expW"]
        curHW = params["curHW"]
        netHW = params["netHW"]
        kldW = params["kldW"]
        bias = params["bias"]
        val = stateValue * stateW + turn * turnW + avgExpansionsPlayed * expW + entropyCurrent * curHW + entropyNetwork * netHW + kldgain * kldW
        val /= 6
        val += bias
        return val > 1

    def wantsToThinkBatch(self, piterators, remainings, currentIterations):

        def entropy(p):
            result = 0
            for idx in range(len(p)):
                if p[idx] > 0:
                    result -= p[idx] * math.log(p[idx])
            return result

        def kldiv(p, q):
            result = 0
            for idx in range(len(p)):
                if q[idx] != 0 and p[idx] != 0:
                    result += p[idx] * math.log2(p[idx] / q[idx])

            return result

        result = []

        for idx, piter in enumerate(piterators):
            if self.maxRemSeen < remainings[idx]:
                self.maxRemSeen = remainings[idx]

            if idx in self.lastIterCount and self.lastIterCount[idx] >= currentIterations[idx] and idx in self.lastIterResults:
                del self.lastIterResults[idx]

            self.lastIterCount[idx] = currentIterations[idx]

            kldgain = 2 * (self.kldgainSum / self.kldgainsSeen)

            presult = piter.getResult()
            now = presult[0]
            netPriors = presult[1]["net_priors"]
            mctsValue = presult[1]["mcts_state_value"]

            if idx in self.lastIterResults:
                prev = self.lastIterResults[idx]
                kldgain = kldiv(prev, now)
                self.kldgainsSeen += 1
                self.kldgainSum += kldgain

            self.lastIterResults[idx] = now

            playedExpansions = self.maxRemSeen - remainings[idx]
            gturn = piter.getGame().getTurn()
            avgPlayedExp = playedExpansions / ((gturn // 2) + 1)

            result.append(self.shouldContinueThinking(piter.getCurrentPlayerParameters(), mctsValue, gturn, avgPlayedExp ,\
                entropy(now), entropy(netPriors), kldgain))
        
        return result

class FixedPlayerAccess(PlayerAccess, metaclass=abc.ABCMeta):
    """
    Very simple version of PlayerAccess: use a fixed configuration
    """

    def __init__(self, parameters):
        self.parameters = parameters

    def getNextMatch(self, runId):
        return (("p1", self.parameters), ("p2", self.parameters))

    def reportResult(self, p1Id, p2Id, winnerId, policyUUID, runId):
        pass

class LeaguePlayerAccess(PlayerAccess, metaclass=abc.ABCMeta):
    """
    evolvoing set of players managed on a server. Starts a thread which in regular intervals pulls the list of players
    from the server, getNextMatch generates matches from that list.
    reportResult pushes results back to the server.
    Requires same sys arguments:
        --command <command server host>
        --secret <server api password>
        --run <run-uuid>
    """

    def __init__(self, activePopulation = 50):
        hasArgs = ("--secret" in sys.argv) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for LeaguePlayerAccess: --secret <server password> and --command <command server host>!")

        self.activePopulation = activePopulation
        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]

        # list of all known players, sorted by their rating in a descending fashion
        # Data format is tuples: (player-id-string, player-rating, player-parameters)
        self.playerList = []

        # reports of games waiting to be send by the network thread.
        self.pendingReports = []

        self.run = None

        self.networkThread = threading.Thread(target=self.procNetwork)
        self.networkThread.daemon = True
        self.networkThread.start()

    def procNetwork(self):
        logMsg("Started league management network thread!")
        while self.run is None:
            time.sleep(1)
        logMsg("League management got run!")
        while True:
            try:
                pendingReportDicts = []
                while len(self.pendingReports) > 0:
                    report = self.pendingReports.pop()
                    rDict = dict()
                    rDict["p1"] = report[0]
                    rDict["p2"] = report[1]
                    rDict["winner"] = report[2]
                    rDict["policy"] = report[3]
                    pendingReportDicts.append(rDict)
                if len(pendingReportDicts) > 0:
                    postJson(self.commandHost + "/api/league/reports/" + self.run, self.secret, pendingReportDicts)

                self.playerList = requestJson(self.commandHost + "/api/league/players/" + self.run, self.secret)

                time.sleep(3)
            except Exception as error:
                print(error)

        logMsg("something bad happened to the league network thread, quitting worker!!!")    
        exit(-1)

    def getNextMatch(self, runId):
        self.run = runId
        while len(self.playerList) < 2:
            logMsg("Waiting for player list to be filled!")
            time.sleep(1)

        # the list might be replaced by the network thread, so take the current list and store the reference
        lst = self.playerList

        maxPIdx = min(self.activePopulation, len(lst)) - 1
        p1Idx = random.randint(0, maxPIdx)
        stepProp = 0.25

        p2Offset = 1
        while True:
            upP2 = p1Idx + p2Offset
            downP2 = p1Idx - p2Offset

            failedUp = upP2 >= len(lst)
            failedDown = downP2 < 0

            if not failedUp and random.random() < stepProp:
                p2Idx = upP2
                break

            if not failedDown and random.random() < stepProp:
                p2Idx = downP2
                break

            if failedDown and failedUp:
                if p1Idx == 0:
                    p2Idx = 1
                else:
                    p2Idx = p1Idx - 1
                break

            p2Offset += 1

        return ((lst[p1Idx][0], lst[p1Idx][2]), (lst[p2Idx][0], lst[p2Idx][2]))

    def reportResult(self, p1Id, p2Id, winnerId, policyUUID, runId):
        self.run = runId
        self.pendingReports.append((p1Id, p2Id, winnerId, policyUUID))

# this self play worker is not supported to be used in local self play!
class LeagueSelfPlayerWorker(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initialState, policy, policyIterator, gameCount, moveDecider,\
            gameReporter, policyUpdater, playerAccess, expansionIncrement, expansionMax,\
            thinkDecider, expansionsMaxSingle):
        self.initialState = initialState
        self.policy = policy
        self.policyIterator = policyIterator
        self.moveDecider = moveDecider
        self.gameReporter = gameReporter
        self.policyUpdater = policyUpdater
        self.playerAccess = playerAccess
        self.expansionIncrement = expansionIncrement
        self.expansionMax = expansionMax
        self.thinkDecider = thinkDecider
        self.expansionsMaxSingle = expansionsMaxSingle

        self.playedBatch = False
        self.initDone = False

        self.prevMoves = 0

        self.gameCount = gameCount

        # the iterators that represent the current state of a game
        self.iterators = []
        # the matches being played at that index of iterators
        self.matches = []
        # tracked frames per game, tracked frames will be reported to be learnt from
        self.tracking = []
        # a list of two element lists, which contains the remaining expansions for each of the players. -> Rests to [max, max] after every game.
        self.remainingForIterations = []
        # a list of numbers, which represent how many expansions have been spent on each game in the current move already. -> Resets to zero after every move.
        self.currentIterationExpansions = []

    def getHDictFor(self, idx):
        match = self.matches[idx]
        hdict = dict()
        hdict[1] = match[0][1]
        hdict[2] = match[1][1]
        return hdict

    def main(self):
        if not "--run" in sys.argv:
            raise Exception("You need to provide arguments for LeaguePlayerAccess: --run <un id>!")
        self.initSelfplay(sys.argv[sys.argv.index("--run")+1])
        while True:
            self.playBatch()

    def initSelfplay(self, runId):
        self.runId = runId
        self.policy = self.policyUpdater.update(self.policy)
        assert not (self.policy is None), "the policy updater returned a None policy!"

        if not self.initDone:
            self.initDone = True
            for _ in range(self.gameCount):
                self.currentIterationExpansions.append(0)
                self.tracking.append([])
                match = self.playerAccess.getNextMatch(self.runId)
                self.matches.append(match)
                hdict = dict()
                hdict[1] = match[0][1]
                hdict[2] = match[1][1]
                self.iterators.append(self.policyIterator.createIterator(self.initialState, hdict))
                self.remainingForIterations.append([self.expansionMax, self.expansionMax])

    def stepIteratorsOnce(self):
        self.policyIterator.iteratePolicyEx(self.policy, self.iterators, iterations = self.expansionIncrement)
        for ix in range(len(self.currentIterationExpansions)):
            self.currentIterationExpansions[ix] += self.expansionIncrement
            curIter = self.iterators[ix]
            curGame = curIter.getGame()
            curPlayer = curGame.getPlayerOnTurnNumber() - 1
            self.remainingForIterations[ix][curPlayer] -= self.expansionIncrement


    def getCurrentRemainingIterations(self):
        result = []
        for ix, rems in zip(self.iterators, self.remainingForIterations):
            pnum = ix.getGame().getPlayerOnTurnNumber()
            result.append(rems[pnum - 1])
                
        return result

    def getMustMove(self):
        curRems = self.getCurrentRemainingIterations()
        wantsIterate = self.thinkDecider.wantsToThinkBatch(self.iterators, curRems, self.currentIterationExpansions)
        isAllowedToIterator = map(lambda x: x[0] >= self.expansionIncrement and x[1] < self.expansionsMaxSingle, zip(curRems, self.currentIterationExpansions))

        return list(map(lambda x: not(x[0] and x[1]), zip(wantsIterate, isAllowedToIterator)))

    def trackFrame(self, gidx, iteratedPolicy, numIterations, remIterations):
        self.tracking[gidx].append([self.iterators[gidx].getGame(), iteratedPolicy, numIterations, remIterations])

    def playMove(self, gidx):
        timeStart = time.monotonic_ns()
        moveDt = 0

        game = self.iterators[gidx].getGame()
        iteratedPolicy = self.iterators[gidx].getResult()
        self.trackFrame(gidx, iteratedPolicy, self.currentIterationExpansions[gidx], self.remainingForIterations[gidx][game.getPlayerOnTurnNumber() - 1])
        moveToPlay = self.moveDecider.decideMove(game, iteratedPolicy[0], iteratedPolicy[1])
        nextGame = game.playMove(moveToPlay)

        moveDt += time.monotonic_ns() - timeStart
        
        if nextGame.hasEnded():
            finishedMatch = self.matches[gidx]
            winnerNumber = nextGame.getWinnerNumber()
            winnerId = None
            if winnerNumber == 1:
                winnerId = finishedMatch[0][0]
            if winnerNumber == 2:
                winnerId = finishedMatch[1][0]
            self.playerAccess.reportResult(finishedMatch[0][0], finishedMatch[1][0], winnerId, self.policy.getUUID(), self.runId)
            self.handleReportFor(gidx, nextGame)

            nextGame = self.initialState
            self.matches[gidx] = self.playerAccess.getNextMatch(self.runId)
            self.remainingForIterations[gidx] = [self.expansionMax, self.expansionMax]

        timeStart = time.monotonic_ns()

        self.currentIterationExpansions[gidx] = 0
        self.remainingForIterations[gidx][nextGame.getPlayerOnTurnNumber() - 1] += self.expansionIncrement
        self.iterators[gidx] = self.policyIterator.createIterator(nextGame, self.getHDictFor(gidx))

        moveDt += time.monotonic_ns() - timeStart

        return moveDt

    def debugPrintState(self, timeNs):
        print("================ Next loop, time spent: %.4fms ->" % (timeNs / 1000000.0))

        gameStrs = []
        for idx, i in enumerate(self.iterators):
            currentExpansions = str(self.currentIterationExpansions[idx])
            remExps = "P1: " + str(self.remainingForIterations[idx][0]) + "\nP2: " + str(self.remainingForIterations[idx][1])
            
            ir = i.getResult()
            iterated = ir[0]
            priors = ir[1]["net_priors"]
            netWins = ir[1]["net_values"]
            gs = i.getGame().prettyString(priors, netWins, iterated, None)

            gameStrs.append("Current: " + currentExpansions + "\n" + remExps + "\n" + gs)

        print(hConcatStrings(gameStrs))


    def playBatch(self):
        """
        playBatch should play one move on every open game.
        Since the number of expansions is dynamic here, this means playBatch should
        play moves until len(self.iterators) number of moves have been tracked, even if they
        happen to be played on the same game.
        """
        
        moveTimeNs = 0

        movesPlayed = self.prevMoves

        if not self.playedBatch:
            self.playedBatch = True
            self.stepIteratorsOnce()

        while movesPlayed < len(self.iterators):

            self.initSelfplay(self.runId)

            moveThinkStart = time.monotonic_ns()
            mustMoves = self.getMustMove()
            moveTimeNs += time.monotonic_ns() - moveThinkStart

            for idx, mustMove in enumerate(mustMoves):
                if mustMove:
                    # the player is out of time, or wants to make a move
                    # -> play a move here for that index
                    moveTimeNs += self.playMove(idx)
                    movesPlayed += 1

            iterStartTime = time.monotonic_ns()

            self.stepIteratorsOnce()

            moveTimeNs += time.monotonic_ns() - iterStartTime

            #Do not enable unless you run a specific config that only has a very low gameCount, else it will spam your terminal.
            #self.debugPrintState(moveTimeNs)

        moveTimeNs /= (movesPlayed - self.prevMoves)

        self.prevMoves = movesPlayed - len(self.iterators)

        moveAvgMs = moveTimeNs / 1000000.0

        return moveAvgMs

    def handleReportFor(self, idx, finalState):
        reports = []

        trackList = self.tracking[idx]

        if len(trackList) == 0:
            return

        result = finalState.getWinnerNumber()
        policyUUID = self.policy.getUUID()

        prevStateUUID = None

        for ti in range(len(trackList)):
            state, iPolicy, numIterations, remIterations = trackList[ti]
            # do not learn from terminal states, there is no move that can be made on them
            assert not state.hasEnded()
            record = dict()
            record["numIterations"] = numIterations
            record["remIterations"] = remIterations
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

        self.tracking[idx] = []

        if len(reports) > 0:
            reports[len(reports) - 1]["final"] = True
            self.gameReporter.reportGame(reports)