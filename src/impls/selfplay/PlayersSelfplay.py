import abc
import uuid
from utils.prints import logMsg
from core.playing.SelfPlayWorker import SelfPlayWorker

from utils.misc import hConcatStrings

import time

import datetime
import numpy as np

import random

# selfplay with "players" (i.e. mcts hyperparameter sets) that compete a in league against each other
# will not support playout caps, maybe they could be added in later, but for now they do not seem to help easily anyway.

class PlayerAccess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getNextMatch(self):
        """
        @return two players to play a game against each other in the form of a tuple of tuples:
        (("player1-ID", {player1}), ("player2-ID", {player2}))
        """

    @abc.abstractmethod
    def reportResult(self, p1Id, p2Id, winnerId):
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
        result = []

        for ci in currentIterations:
            diff = random.random() * 100 - 50            
            result.append(self.expansions > ci - diff)

        return result

        #return [self.expansions > ci for ci in currentIterations]

class FixedPlayerAccess(PlayerAccess, metaclass=abc.ABCMeta):
    """
    Very simple version of PlayerAccess: use a fixed configuration
    """

    def __init__(self, parameters):
        self.parameters = parameters

    def getNextMatch(self):
        return (("p1", self.parameters), ("p2", self.parameters))

    def reportResult(self, p1Id, p2Id, winnerId):
        print("Reported result:", p1Id, "vs", p2Id, "; winner is:", winnerId)

class LeagueSelfPlayerWorker(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initialState, policy, policyIterator, gameCount, moveDecider,\
            gameReporter, policyUpdater, playerAccess, expansionIncrement, expansionMax,\
            thinkDecider):
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

        self.prevMoves = 0

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
        for _ in range(gameCount):
            self.currentIterationExpansions.append(0)
            self.tracking.append([])
            match = self.playerAccess.getNextMatch()
            self.matches.append(match)
            hdict = dict()
            hdict[1] = match[0][1]
            hdict[2] = match[1][1]
            self.iterators.append(self.policyIterator.createIterator(initialState, hdict))
            self.remainingForIterations.append([self.expansionMax, self.expansionMax])

    def getHDictFor(self, idx):
        match = self.matches[idx]
        hdict = dict()
        hdict[1] = match[0][1]
        hdict[2] = match[1][1]
        return hdict

    def main(self):
        self.initSelfplay()
        while True:
            self.playBatch()

    def initSelfplay(self):
        self.policy = self.policyUpdater.update(self.policy)
        assert not (self.policy is None), "the policy updater returned a None policy!"

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
        isAllowedToIterator = map(lambda x: x >= self.expansionIncrement, curRems)

        return list(map(lambda x: not(x[0] and x[1]), zip(wantsIterate, isAllowedToIterator)))

    def trackFrame(self, gidx, iteratedPolicy):
        self.tracking[gidx].append([self.iterators[gidx].getGame(), iteratedPolicy])

    def playMove(self, gidx):
        timeStart = time.monotonic_ns()
        moveDt = 0

        iteratedPolicy = self.iterators[gidx].getResult()
        self.trackFrame(gidx, iteratedPolicy)
        game = self.iterators[gidx].getGame()
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
            self.playerAccess.reportResult(finishedMatch[0][0], finishedMatch[1][0], winnerId)
            self.handleReportFor(gidx, nextGame)

            nextGame = self.initialState
            self.matches[gidx] = self.playerAccess.getNextMatch()
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

        while movesPlayed < len(self.iterators):

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
            self.debugPrintState(moveTimeNs)

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

        self.tracking[idx] = []

        if len(reports) > 0:
            reports[len(reports) - 1]["final"] = True
            self.gameReporter.reportGame(reports)