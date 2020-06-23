import abc
import uuid
import datetime
from utils.prints import logMsg
from core.playing.SelfPlayWorker import SelfPlayWorker


from utils.req import requestJson, postJson, requestBytes, postBytes
from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

import numpy as np

import time

import subprocess
import sys

import socket

import uuid

import signal
import ctypes
libc = ctypes.CDLL("libc.so.6")
def set_pdeathsig(sig = signal.SIGTERM):
    def callable():
        return libc.prctl(1, sig)
    return callable

class EvaluationAccess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def requestEvaluation(self, gamesPackage):
        """
        push a package to be evaluated. Expects an UUID to identify the evaluation task
        """

    @abc.abstractmethod
    def pollEvaluationResults(self):
        """
        return any finished evaluations to be used by the caller. No result is ever returned twice!
        Format: dict() uuid -> ([iterationResult], networkId)
        iterationResult is a tuple out of a policy iterator getResults()
        """

class RemoteEvaluationAccess(EvaluationAccess, metaclass=abc.ABCMeta):
    def __init__(self):
        hasArgs = "--command" in sys.argv

        if not hasArgs:
            raise Exception("You need to provide arguments for LocalEvaluationAccess: --command <command server host>!")

        self.evalHost = sys.argv[sys.argv.index("--command")+1].replace("https", "http")
        self.evalHost += ":4242"

    def requestEvaluation(self, gamesPackage):
        """
        push a package to be evaluated. Expects an UUID to identify the evaluation task
        """
        return postBytes(self.evalHost + "/queue", "", encodeToBson([g.store() for g in gamesPackage]), expectResponse=True)

    def pollEvaluationResults(self):
        """
        return any finished evaluations to be used by the caller. No result is ever returned twice!
        Format: dict() uuid -> ([iterationResult], networkId)
        iterationResult is a tuple out of a policy iterator getResults()
        """
        workResults = requestJson(self.evalHost + "/results", "")
        if len(workResults) == 0:
            time.sleep(0.5)
        result = dict()
        for rId in workResults:
            rBytes = requestBytes(self.evalHost + "/results/" + rId, "")
            rDict = decodeFromBson(rBytes)
            result[rId] = (rDict["iterations"], rDict["network"])
        return result

class LocalEvaluationAccess(EvaluationAccess, metaclass=abc.ABCMeta):
    """
    Run a local evaluation server and a number of local EvaluationWorkers. 
    """
    def __init__(self, workerN=1):
        logMsg("Starting local eval_manager!")
        subprocess.Popen(["node", "eval_manager/server.js"], preexec_fn = set_pdeathsig(signal.SIGTERM))
        hasArgs = "--command" in sys.argv

        if not hasArgs:
            raise Exception("You need to provide arguments for LocalEvaluationAccess: --command <command server host>!")

        self.commandHost = sys.argv[sys.argv.index("--command")+1]
        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.run = sys.argv[sys.argv.index("--run")+1]

        self.evalHost = "http://127.0.0.1:4242"

        for w in range(workerN):
            logMsg("Starting local worker %i!" % (w + 1))
            subprocess.Popen(["python", "-m", "core.mains.distributed", "--command", self.commandHost, "--secret", self.secret, "--run", self.run, "--evalserver", self.evalHost, "--eval", socket.gethostname(), "--windex", str(w+1)], preexec_fn = set_pdeathsig(signal.SIGTERM))

        time.sleep(3)

    def requestEvaluation(self, gamesPackage):
        """
        push a package to be evaluated. Expects an UUID to identify the evaluation task
        """
        return postBytes(self.evalHost + "/queue", "", encodeToBson([g.store() for g in gamesPackage]), expectResponse=True)

    def pollEvaluationResults(self):
        """
        return any finished evaluations to be used by the caller. No result is ever returned twice!
        Format: dict() uuid -> ([iterationResult], networkId)
        iterationResult is a tuple out of a policy iterator getResults()
        """
        workResults = requestJson(self.evalHost + "/results", "")
        if len(workResults) == 0:
            time.sleep(0.5)
        result = dict()
        for rId in workResults:
            rBytes = requestBytes(self.evalHost + "/results/" + rId, "")
            rDict = decodeFromBson(rBytes)
            result[rId] = (rDict["iterations"], rDict["network"])
        return result

class FakeEvaluationAccess(EvaluationAccess, metaclass=abc.ABCMeta):
    """
    Implementation for testing purposes, does not actually run any policy, just produces dirichlet noise "evaluations".
    Begins a new "iteration" after a configured number of evaluations.
    """

    def __init__(self, iterationSize):
        self.pendingPolls = dict()
        self.iterationSize = iterationSize
        self.currentIterationProcessed = 0
        self.currentFakeNetworkId = str(uuid.uuid4())
        # self.respondedStates = set()

    def requestEvaluation(self, gamesPackage):
        """
        push a package to be evaluated. Expects an UUID to identify the evaluation task
        """
        pollId = str(uuid.uuid4())
        results = []
        for idx, game in enumerate(gamesPackage):
            # can (should) happen on the next iteration
            #assert not game in self.respondedStates, "Duplicate evaluation: " + str(game)
            # self.respondedStates.add(game)

            randomPolicy = np.random.dirichlet([0.8] * game.getMoveCount()).astype(np.float)
            generics = dict()
            generics["net_values"] = np.random.dirichlet([0.8] * (game.getPlayerCount() + 1)).astype(np.float)
            generics["net_priors"] = np.random.dirichlet([0.8] * game.getMoveCount()).astype(np.float)
            generics["mcts_state_value"] = 0
            results.append((randomPolicy, generics))
        self.pendingPolls[pollId] = (results, self.currentFakeNetworkId)

        self.currentIterationProcessed += len(gamesPackage)

        return pollId

    def pollEvaluationResults(self):
        """
        return any finished evaluations to be used by the caller. No result is ever returned twice!
        Format: dict() uuid -> ([results], networkId)
        """
        result = self.pendingPolls
        self.pendingPolls = dict()

        if self.currentIterationProcessed > self.iterationSize:
            logMsg("Faking a new iteration!")
            self.currentIterationProcessed = 0
            self.currentFakeNetworkId = str(uuid.uuid4())

        return result

def noBiasArgMax(ar):
    if len(ar) == 0:
        return -1

    offsetIdx = np.random.randint(0, len(ar))
    bestIdx = -1
    bestValue = -1
    for baseIdx in range(len(ar)):
        idx = (baseIdx + offsetIdx) % len(ar)
        if bestIdx == -1 or bestValue < ar[idx]:
            bestIdx = idx
            bestValue = ar[idx]

    return bestIdx

class MCTSNode():
    """
    A MCTS node for MCTS self-play. Handles transpositions.
    """

    def __init__(self, state, nodes = dict()):
        self.state = state

        # dict of other nodes, game state -> node. Use to find children when moving down the tree. "Simple way" of handling transpositions.
        # backups still only go back the path that it walked down and ignore cases of multiple parents.
        self.nodes = nodes

        # not expanded yet
        self.isExpanded = False
        self.legalMoveKeys = None
        self.edgeVisits = None
        self.edgePriors = None

        self.virtualLosses = 0

        self.observedResults = np.zeros(self.state.getPlayerCount() + 1)

        self.terminalResult = None

    def _getVirtualLossForEdgeTarget(self, moveKey):
        targetState = self.state.playMove(moveKey)
        if targetState in self.nodes:
            return self.nodes[targetState].virtualLosses
        else:
            return 0

    def _pickMove(self, cpuct, fpu):
        vlossCnt = 0
        vlosses = dict()
        for i, mkey in enumerate(self.legalMoveKeys):
            vlosses[i] = self._getVirtualLossForEdgeTarget(mkey)
            vlossCnt += vlosses[i]

        # + .00001 means that in the case of a new node with zero visits it will chose whatever has the best P
        # instead of just the move with index 0 (for a constant FPU at least)
        # but there is little effect in other cases
        visitsRoot = (np.sum(self.edgeVisits) + vlossCnt) ** 0.5 + 0.00001

        valuesTmp = np.zeros_like(self.legalMoveKeys, dtype=np.float)
        
        # if self.state.getTurn() == 6:
        #     print(str(self.state))

        for i, mkey in enumerate(self.legalMoveKeys):
            edgeCombinedVisits = self.edgeVisits[i] + vlosses[i]
            if edgeCombinedVisits == 0:
                nodeQ = fpu
            else:
                nodeQ = self.edgeTotalValues[i] / edgeCombinedVisits

            nodeU = self.edgePriors[i] * (visitsRoot / (1.0 + edgeCombinedVisits))

            # if self.state.getTurn() == 6:
            #     print(self.legalMoveKeys[mkey], "=>", nodeQ, "TV %.4f" % self.edgeTotalValues[i], "EV", self.edgeVisits[i], "VL", vlosses[i], "Priors %.4f" % self.edgePriors[i])

            valuesTmp[i] = nodeQ + cpuct * nodeU

        return noBiasArgMax(valuesTmp)

    def expand(self, movePriors, generics, networkId):
        """
        Fill in network evaluations
        """

        self.rawPriors = movePriors
        self.networkId = networkId
        self.generics = generics

        self.legalMoveKeys = np.array(self.state.getLegalMoves())

        self.edgeVisits = np.zeros_like(self.legalMoveKeys, dtype=np.int32)
        self.edgePriors = np.zeros_like(self.legalMoveKeys, dtype=np.float)
        self.edgeTotalValues = np.zeros_like(self.legalMoveKeys, dtype=np.float)

        for i, mkey in enumerate(self.legalMoveKeys):
            self.edgePriors[i] = movePriors[mkey]
        
        self.edgePriors /= np.sum(self.edgePriors)

        self.isExpanded = True

    def selectDown(self, cpuct, fpu):
        node = self
        passedNodes = []

        while node.isExpanded and not node.state.hasEnded():
            nextMove = node._pickMove(cpuct, fpu)
            passedNodes.append((node, nextMove))

            nextState = node.state.playMove(node.legalMoveKeys[nextMove])
            if nextState in self.nodes:
                node = self.nodes[nextState]
            else:
                node = MCTSNode(nextState, self.nodes)
                self.nodes[nextState] = node

        failNode = node

        for node, move in passedNodes[1:]:
            node.virtualLosses += 1

        if not failNode.state.hasEnded():
            failNode.virtualLosses += 1

        return passedNodes, failNode

    def backup(self, forMove, values, drawValue):
        """
        backup the given values found at this location using the specific drawValue.
        Does not call backup on parents, the user need to remember the path and call backup on each node directly.
        """
        self.edgeVisits[forMove] += 1
        self.edgeTotalValues[forMove] += values[self.state.getPlayerOnTurnNumber()] + values[0] * drawValue

        self.observedResults += values

    def getWinnerArray(self):
        # encoding results as a list of game results was not a great idea. However I do not want to change the code that expects that data format now
        # so fake out arrays like that, but limit the length somewhat...
        result = []
        if np.sum(self.observedResults) > 50:
            norm = self.observedResults / np.sum(self.observedResults)
            for i in range(len(norm)):
                result += [i] * int(norm[i] * 50)
        else:
            for i in range(len(self.observedResults)):
                result += [i] * int(self.observedResults[i])
        return result

    def removeVirtualLoss(self):
        if not self.state.hasEnded():
            self.virtualLosses -= 1
        assert self.virtualLosses >= 0

    def getTerminalResult(self):
        if self.terminalResult is None:
            numOutputs = self.state.getPlayerCount() + 1
            r = [0] * numOutputs
            winner = self.state.getWinnerNumber()
            r[winner] = 1
            self.terminalResult = np.array(r, dtype=np.float32)

        return self.terminalResult

    def getMoveDistribution(self):
        result = np.zeros(self.state.getMoveCount(), dtype=np.float32)
        for i in range(len(self.legalMoveKeys)):
            result[self.legalMoveKeys[i]] = self.edgeVisits[i]
        result /= float(np.sum(self.edgeVisits))
        return result

# standing for the policy ID of the first iteration
fooUUID = str(uuid.uuid4())

def packageReport(state, iPolicy, winners, nextPolicy, policyUUID):
    global fooUUID

    # do not learn from terminal states, there is no move that can be made on them
    assert not state.hasEnded()
    record = dict()
    record["gameCtor"] = state.getGameConstructorName()
    record["gameParams"] = state.getGameConstructorParams()
    record["knownResults"] = winners
    record["generics"] = dict(iPolicy[1])
    record["policyIterated"] = iPolicy[0]
    record["uuid"] = str(uuid.uuid4())
    if policyUUID is None:
        policyUUID = fooUUID
    record["policyUUID"] = policyUUID
    record["state"] = state.store()
    record["gamename"] = state.getGameName()
    record["creation"] = datetime.datetime.utcnow().timestamp()
    record["final"] = False

    # if the game is over after the current move is played, encode that as a uniform distribution over all moves
    reply = np.ones_like(record["policyIterated"])
    reply /= reply.shape[0]
    if nextPolicy is not None:
        # if however there is another move played, track the distribution for that move
        reply = nextPolicy
    record["reply"] = reply

    return record

class SelfPlayTree():
    def __init__(self, initialState, maxPackageSize, maxPendingPackages, evalAccess, cpuct, fpu, drawValue):
        self.initialState = initialState
        self.maxPackageSize = maxPackageSize
        self.maxPendingPackages = maxPendingPackages
        self.evalAccess = evalAccess
        self.cpuct = cpuct
        self.fpu = fpu
        self.drawValue = drawValue

        # UUIDs handled by this tree
        self.pendingEvalIds = set()

        # (UUID, N) -> (MCTSNode, paths)
        self.pendingEvals = dict()

        # game state -> (UUID, N)
        self.pendingEvalsInverse = dict()

        # doubles every time a package is submitted, but not beyond self.maxPackageSize
        self.currentMaxPackageSize = 2
        # doubles every time a package is submitted and currentMaxPackageSize cannot increase anymore
        # does not grow beyond self.maxPendingPackages
        self.currentMaxPendingPackages = 2

        self.root = MCTSNode(initialState)

        self.reportedStates = set()
        self.pendingReports = []

        self.newPendings = dict()

        self.allowNewGames = True

    def backup(self, nodes, result):
        """
        nodes is a list of non-terminal nodes which end with the given result.
        """

        for node, move in nodes:
            node.backup(move, result, self.drawValue)

        # the root node does not get virtual losses, they would not have an effect anyway. You have to visit it always.
        for node, move in nodes[1:]:
            node.removeVirtualLoss()

        for nidx, (node, move) in enumerate(nodes):
            iPolicy = (node.rawPriors, node.generics)
            policyUUID = node.networkId

            nextPolicy = None
            if (nidx + 1) < len(nodes):
                nextPolicy = nodes[nidx + 1][0].rawPriors
            
            record = packageReport(node.state, iPolicy, node.getWinnerArray(), nextPolicy, policyUUID)
            self.pendingReports.append(record)

        self.pendingReports[-1]["final"] = True

    def pollReports(self):
        ret = self.pendingReports
        self.pendingReports = []
        return ret
        
    def handleSelection(self, passedNodes, failNode):
        if failNode.state.hasEnded():
            self.backup(passedNodes, failNode.getTerminalResult())
        elif failNode.state in self.pendingEvalsInverse:
            evalId = self.pendingEvalsInverse[failNode.state]
            pendEval = self.pendingEvals[evalId]
            self.pendingEvals[evalId] = (pendEval[0], pendEval[1] + [passedNodes])
        elif failNode.state in self.newPendings:
            self.newPendings[failNode.state] = (failNode, self.newPendings[failNode.state][1] + [passedNodes])
        else:
            self.newPendings[failNode.state] = (failNode, [passedNodes])

    def incPackageSizes(self):
        self.currentMaxPackageSize *= 2
        if self.currentMaxPackageSize > self.maxPackageSize:
            self.currentMaxPackageSize = self.maxPackageSize
            self.currentMaxPendingPackages *= 2
            if self.currentMaxPendingPackages > self.maxPendingPackages:
                self.currentMaxPendingPackages = self.maxPendingPackages

        logMsg("Limits are now: %i package size, %i packages" % (self.currentMaxPackageSize, self.currentMaxPendingPackages))

    def queuePendings(self, nextPending):
        package = list(dict.keys(nextPending))

        assert len(package) <= self.currentMaxPackageSize

        logMsg("Queue a new package of size %i" % len(package))

        packageId = self.evalAccess.requestEvaluation(package)
        self.pendingEvalIds.add(packageId)

        for idx, g in enumerate(package):
            evalId = (packageId, idx)
            self.pendingEvalsInverse[g] = evalId
            self.pendingEvals[evalId] = (nextPending[g][0], nextPending[g][1])

    def handleNewPendings(self):
        if len(self.newPendings) > 0:
            while len(self.newPendings) >= self.currentMaxPackageSize:
                nextPendings = dict()
                pkeys = list(dict.keys(self.newPendings))[:self.currentMaxPackageSize]
                for pk in pkeys:
                    nextPendings[pk] = self.newPendings[pk]
                    del self.newPendings[pk]

                self.queuePendings(nextPendings)

            self.beginGames()

            if len(self.newPendings) > 0:
                if len(self.pendingEvalIds) == 0:
                    self.queuePendings(self.newPendings)
                    self.newPendings = dict()
                else:
                    logMsg("Will wait to fill current package, has size %i, needs %i!" % (len(self.newPendings), self.currentMaxPackageSize))

            logMsg("Now waiting for %i packages with %i games!" % (len(self.pendingEvalIds), self.countPendingGames()))

    def countPendingGames(self):
        cnt = 0
        for n, ps in dict.values(self.pendingEvals):
            cnt += len(ps)

        for n, ps in dict.values(self.newPendings):
            cnt += len(ps)

        return cnt

    def fillPendingWithNews(self):
        lastSize = -1
        failCnt = 0
        cnt = 0
        while len(self.newPendings) < self.currentMaxPackageSize and self.countPendingGames() < self.currentMaxPackageSize * self.currentMaxPendingPackages:
            if lastSize == len(self.newPendings):
                failCnt += 1
            else:
                failCnt = 0
            if failCnt > 50:
                break
            lastSize = len(self.newPendings)
            passedNodes, failNode = self.root.selectDown(self.cpuct, self.fpu)
            cnt += 1
            self.handleSelection(passedNodes, failNode)
        return cnt

    def beginGames(self):
        while self.allowNewGames and len(self.pendingEvalIds) < self.currentMaxPendingPackages and self.countPendingGames() < self.currentMaxPackageSize * self.currentMaxPendingPackages:
            newGames = self.fillPendingWithNews()
            if newGames == 0:
                logMsg("Cannot start more games!")
                break
            logMsg("Begin %i new games!" % newGames)
            self.handleNewPendings()

    def hasOpenGames(self):
        return self.countPendingGames() > 0

    def continueGames(self, evalResults):
        self.incPackageSizes()

        for uuid in evalResults:
            if not (uuid in self.pendingEvalIds):
                continue
            self.pendingEvalIds.remove(uuid)

            results, network = evalResults[uuid]
            pendingContinuations = []
            for idx, (policy, generics) in enumerate(results):
                evalId = (uuid, idx)
                node, paths = self.pendingEvals[evalId]
                node.expand(policy, generics, network)
                for path in paths:
                    pendingContinuations.append((node, path))
                del self.pendingEvals[evalId]
                del self.pendingEvalsInverse[node.state]

            preSize = len(self.newPendings)

            for node, path in pendingContinuations:
                passedNodes, failNode = node.selectDown(self.cpuct, self.fpu)
                self.handleSelection(path + passedNodes, failNode)

            postSize = len(self.newPendings)

            logMsg("Continue %i games from %i positions to %i positions" % (len(pendingContinuations), len(results), postSize - preSize))

            self.handleNewPendings()


# there should only be one single of these, ran on a controlled machine.
# evaluation workers need to be started elsewhere and communicate via the evaluation-manager server process (hidden behind evalAccess implementation)
# with the TreeSelfPlayWorker. In frametime eval mode that has to be local, too. => make the frametime eval code patch the evalAccess implementation used.
class TreeSelfPlayWorker(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initialState, maxPackageSize, maxPendingPackages, gameReporter, evalAccess, cpuct, fpu, drawValue):
        self.initialState = initialState
        self.maxPackageSize = maxPackageSize
        self.maxPendingPackages = maxPendingPackages
        self.gameReporter = gameReporter
        self.evalAccess = evalAccess
        self.cpuct = cpuct
        self.fpu = fpu
        self.drawValue = drawValue

        self.prevTrees = []
        self.currentTree = self.newSelfPlayTree()
        self.seenNetworks = set()

    def newSelfPlayTree(self):
        return SelfPlayTree(self.initialState, self.maxPackageSize, self.maxPendingPackages,\
                    self.evalAccess, self.cpuct, self.fpu, self.drawValue)

    def main(self):
        while True:
            self.playBatch()

    def initSelfplay(self, runId):
        pass

    def playBatch(self):
        newReports = []

        startTime = time.monotonic_ns()

        while len(newReports) == 0:
            self.currentTree.beginGames()
            evalResults = dict()
            while len(evalResults) == 0:
                evalResults = self.evalAccess.pollEvaluationResults()
            self.currentTree.continueGames(evalResults)

            newReports += self.currentTree.pollReports()

            for prevTree in self.prevTrees:
                prevTree.continueGames(evalResults)
                newReports += prevTree.pollReports()

            prevTreeCount = len(self.prevTrees)
            self.prevTrees = list(filter(lambda x: x.hasOpenGames(), self.prevTrees))

            if len(self.prevTrees) < prevTreeCount:
                logMsg("=========================================================================")
                logMsg("=========================================================================")
                logMsg("Completed %i prev trees!" % (prevTreeCount - len(self.prevTrees)))
                logMsg("=========================================================================")
                logMsg("=========================================================================")

            if len(self.prevTrees) > 0:
                gsum = 0
                for pt in self.prevTrees:
                    gsum += pt.countPendingGames()
                logMsg("Previous trees are still playing out %i games!" % gsum)

            newIteration = False
            for uuid in evalResults:
                results, network = evalResults[uuid]
                if network is not None and not (network in self.seenNetworks):
                    self.seenNetworks.add(network)
                    newIteration = True
            
            print(self.seenNetworks)

            if newIteration:
                logMsg("=========================================================================")
                logMsg("=========================================================================")
                logMsg("======================= Detected a new iteration! =======================")
                logMsg("=========================================================================")
                logMsg("=========================================================================")
                logMsg("Known networks: ", self.seenNetworks)
                self.currentTree.allowNewGames = False
                self.prevTrees.append(self.currentTree)
                if len(self.prevTrees) > 1:
                    logMsg("WARN: there are multiple prev trees: %i" % len(self.prevTrees))
                self.currentTree = self.newSelfPlayTree()


        self.gameReporter.reportGame(newReports)

        dt = (time.monotonic_ns() - startTime) / 1000000.0

        return (dt / len(newReports)), None
