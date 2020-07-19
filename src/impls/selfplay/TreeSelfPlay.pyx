# cython: profile=False

import abc
import uuid
import datetime
from utils.prints import logMsg
from core.playing.SelfPlayWorker import SelfPlayWorker


from utils.req import requestJson, postJson, requestBytes, postBytes
from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

import numpy as np

import json

import time

import subprocess
import sys

import socket

import uuid
import setproctitle

import multiprocessing as mp

from utils.fields.fields cimport mallocWithZero
from libc.stdlib cimport free

import random

from libc.stdlib cimport rand, RAND_MAX

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
        Format: dict() uuid -> ([iterationResult], networkId, workerName)
        iterationResult is a tuple out of a policy iterator getResults()
        """

def postBytesWithEncode(url, data):
    postBytes(url, "", encodeToBson(data))

def pollResults(evalHost, queue):
    setproctitle.setproctitle("x0_poll_eval_results")
    while True:
        workResults = requestJson(evalHost + "/results", "")
        if len(workResults) == 0:
            time.sleep(0.5)
            continue
        result = dict()
        for rId in workResults:
            rBytes = requestBytes(evalHost + "/results/" + rId, "")
            rDict = decodeFromBson(rBytes)
            result[rId] = (rDict["iterations"], rDict["network"], rDict["workerName"])
        queue.put(result)

def rolloutPosition(object root, int n):
    results = np.zeros((root.getPlayerCount() + 1), dtype=np.float32)

    cdef int idx

    for idx in range(n):
        position = root
        while not position.hasEnded():
            move = position.getRandomLegalMove()
            position = position.playMove(move)
        results[position.getWinnerNumber()] += 1

    results /= np.sum(results)

    return results

def rolloutWithUniformMCTS(object root, int n):
    results = np.zeros((root.getPlayerCount() + 1), dtype=np.float32)
    tree = MCTSNode(root)

    cdef int idx

    for idx in range(n):
        passedNodes, failNode = tree.selectDown(1, 0, 99, 0)

        if failNode.state.hasEnded():
            result = failNode.getTerminalResult()
            results[failNode.state.getWinnerNumber()] += 1
            for node, move in passedNodes:
                node.backup(move, result, 0.5)
        else:
            assert not failNode.isExpanded
            
            r = np.random.rand(failNode.state.getMoveCount())
            r /= np.sum(r)
            failNode.expand(r, None, "A", 0, "")

    return results

def tryMCTSRollout(object root, int n, int minResults, int alternativeRollouts):
    x = rolloutWithUniformMCTS(root, n)
    xs = np.sum(x)
    if xs < minResults:
        x = rolloutPosition(root, alternativeRollouts)
    else:
        x /= xs
    return x

class RemoteEvaluationAccess(EvaluationAccess, metaclass=abc.ABCMeta):
    def __init__(self, workers=4):
        hasArgs = "--command" in sys.argv

        if not hasArgs:
            raise Exception("You need to provide arguments for RemoteEvaluationAccess: --command <command server host>!")

        self.evalHost = sys.argv[sys.argv.index("--command")+1].replace("https", "http").replace(":8042", "")
        self.evalHost += ":4242"
        logMsg("Remote evaluation access will use server %s" % self.evalHost)

        self.pool = mp.Pool(processes=workers)
        self.queue = mp.Queue(maxsize=5000)
        self.proc = mp.Process(target=pollResults, args=(self.evalHost, self.queue))
        self.proc.start()

    def requestEvaluation(self, gamesPackage):
        """
        push a package to be evaluated. Expects an UUID to identify the evaluation task
        """
        myId = str(uuid.uuid4())
        self.pool.apply_async(postBytesWithEncode, (self.evalHost + "/queue/" + myId, [g.store() for g in gamesPackage]))
        return myId

    def pollEvaluationResults(self):
        """
        return any finished evaluations to be used by the caller. No result is ever returned twice!
        Format: dict() uuid -> ([iterationResult], networkId, workerName)
        iterationResult is a tuple out of a policy iterator getResults()
        """
        result = dict()
        try:
            qdict = self.queue.get_nowait()
            for qkey in qdict:
                result[qkey] = qdict[qkey]
        except:
            pass
        
        if len(result) == 0:
            time.sleep(0.1)

        return result

class LocalEvaluationAccess(EvaluationAccess, metaclass=abc.ABCMeta):
    """
    Run a local evaluation server and a number of local EvaluationWorkers. 
    forceRun and forceCfg are meant to be used by the frametime evaluator code only.
    """
    def __init__(self, workerN=1, forceRun=None, forceCfg=None):
        logMsg("Starting local eval_manager!")
        subprocess.Popen(["node", "eval_manager/server.js"], preexec_fn = set_pdeathsig(signal.SIGTERM))
        hasArgs = "--command" in sys.argv

        time.sleep(1)

        if not hasArgs:
            raise Exception("You need to provide arguments for LocalEvaluationAccess: --command <command server host>!")

        self.commandHost = sys.argv[sys.argv.index("--command")+1]
        self.secret = sys.argv[sys.argv.index("--secret")+1]
        if forceRun is not None:
            self.run = forceRun
            logMsg("LocalEvaluationAccess is forced to use run %s" % forceRun)
        else:
            self.run = sys.argv[sys.argv.index("--run")+1]

        self.evalHost = "http://127.0.0.1:4242"

        myHostname = socket.gethostname()
        for w in range(workerN):
            logMsg("Starting local worker %i!" % (w + 1))
            wparams = ["python", "-m", "core.mains.distributed", "--command", self.commandHost, "--secret", self.secret, "--run", self.run, "--evalserver", self.evalHost, "--eval", myHostname + str(w + 1), "--windex", str(w+1)]
            if forceCfg is not None:
                wparams += ["--fconfig", forceCfg]
            subprocess.Popen(wparams, preexec_fn = set_pdeathsig(signal.SIGTERM))
            time.sleep(1)

        self.pool = mp.Pool(processes=2)

        time.sleep(3)

    def requestEvaluation(self, gamesPackage):
        """
        push a package to be evaluated. Expects an UUID to identify the evaluation task
        """
        myId = str(uuid.uuid4())
        self.pool.apply_async(postBytesWithEncode, (self.evalHost + "/queue/" + myId, [g.store() for g in gamesPackage]))
        return myId

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
            result[rId] = (rDict["iterations"], rDict["network"], rDict["workerName"])
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

            randomPolicy = np.random.dirichlet([0.8] * game.getMoveCount()).astype(np.float32)
            generics = dict()
            generics["net_values"] = np.random.dirichlet([0.8] * (game.getPlayerCount() + 1)).astype(np.float32)
            generics["net_priors"] = np.random.dirichlet([0.8] * game.getMoveCount()).astype(np.float32)
            generics["mcts_state_value"] = 0
            results.append((randomPolicy, generics))
        self.pendingPolls[pollId] = (results, self.currentFakeNetworkId, "fake")

        self.currentIterationProcessed += len(gamesPackage)

        return pollId

    def pollEvaluationResults(self):
        """
        return any finished evaluations to be used by the caller. No result is ever returned twice!
        Format: dict() uuid -> ([results], networkId, workerName)
        """
        result = self.pendingPolls
        self.pendingPolls = dict()

        if self.currentIterationProcessed > self.iterationSize:
            logMsg("Faking a new iteration!")
            self.currentIterationProcessed = 0
            self.currentFakeNetworkId = str(uuid.uuid4())

        return result

def noBiasArgMax(float[:] ar):
    if len(ar) == 0:
        return -1

    cdef int offsetIdx = np.random.randint(0, len(ar))
    cdef int bestIdx = -1
    cdef float bestValue = -1
    cdef int baseIdx
    cdef int idx

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

    def __init__(self, state, nodes = None):
        self.state = state

        # dict of other nodes, game state -> node. Use to find children when moving down the tree. "Simple way" of handling transpositions.
        # backups still only go back the path that it walked down and ignore cases of multiple parents.
        self.nodes = nodes
        if self.nodes is None:
            self.nodes = dict()

        # not expanded yet
        self.isExpanded = 0
        self.legalMoveKeys = None
        self.edgeVisits = None
        self.edgePriors = None

        self.virtualLosses = 0

        self.terminalResult = None

        self.reportCount = 0

        self.childStates = dict()
        self.childNodes = dict()

        self.valuesTmp = None

        self.networkIteration = -1

    def getChildren(self):
        result = dict()
        cdef int mkey
        if self.legalMoveKeys is not None:
            for mkey in self.legalMoveKeys:
                targetState = self.state.playMove(mkey)
                if targetState in self.nodes:
                    result[mkey] = self.nodes[targetState]
        return result

    def _getVirtualLossForEdgeTarget(self, moveKey):
        if moveKey in self.childNodes:
            return self.childNodes[moveKey].virtualLosses
        else:
            targetState = self.childStates[moveKey]
            if targetState in self.nodes:
                self.childNodes[moveKey] = self.nodes[targetState]
                return self.childNodes[moveKey].virtualLosses
            else:
                return 0

    def _pickMove(self, float cpuct, float fpu):
        cdef int vlossCnt = 0
        cdef int* vlosses = <int*> mallocWithZero(len(self.legalMoveKeys) * sizeof(int))

        cdef int i
        cdef int mkey

        for i, mkey in enumerate(self.legalMoveKeys):
            vlosses[i] = self._getVirtualLossForEdgeTarget(mkey)
            vlossCnt += vlosses[i]

        # + .00001 means that in the case of a new node with zero visits it will chose whatever has the best P
        # instead of just the move with index 0 (for a constant FPU at least)
        # but there is little effect in other cases
        cdef float visitsRoot = (np.sum(self.edgeVisits) + vlossCnt) ** 0.5 + 0.00001
        
        # if self.state.getTurn() == 6:
        #     print(str(self.state))

        cdef int[:] edgeVisits = self.edgeVisits
        cdef float[:] edgeTotalValues = self.edgeTotalValues
        cdef float[:] edgePriors = self.edgePriors
        cdef float[:] valuesTmp = self.valuesTmp

        cdef float nodeQ, nodeU
        cdef int edgeCombinedVisits 

        for i, mkey in enumerate(self.legalMoveKeys):
            edgeCombinedVisits = edgeVisits[i] + vlosses[i]
            if edgeCombinedVisits == 0:
                nodeQ = fpu
            else:
                nodeQ = edgeTotalValues[i] / edgeCombinedVisits

            nodeU = edgePriors[i] * (visitsRoot / (1.0 + edgeCombinedVisits))

            # if self.state.getTurn() == 6:
            #     print(self.legalMoveKeys[mkey], "=>", nodeQ, "TV %.4f" % self.edgeTotalValues[i], "EV", self.edgeVisits[i], "VL", vlosses[i], "Priors %.4f" % self.edgePriors[i])

            valuesTmp[i] = nodeQ + cpuct * nodeU

        free(vlosses)

        pickedMove = noBiasArgMax(valuesTmp)

        # if np.sum(self.edgeVisits) == 0 and self.state.getTurn() == 6:
        #     print("!!!", pickedMove, self.edgePriors, self.legalMoveKeys)

        return pickedMove

    def expand(self, movePriors, generics, networkId, networkIteration, workerName):
        """
        Fill in network evaluations
        """

        self.rawPriors = movePriors
        self.networkId = networkId
        self.networkIteration = networkIteration
        self.generics = generics
        self.workerName = workerName

        self.legalMoveKeys = np.array(self.state.getLegalMoves())

        self.valuesTmp = np.zeros_like(self.legalMoveKeys, dtype=np.float32)

        self.edgeVisits = np.zeros_like(self.legalMoveKeys, dtype=np.int32)
        self.edgePriors = np.zeros_like(self.legalMoveKeys, dtype=np.float32)
        self.edgeTotalValues = np.zeros_like(self.legalMoveKeys, dtype=np.float32)

        cdef int i, mkey

        for i, mkey in enumerate(self.legalMoveKeys):
            self.edgePriors[i] = movePriors[mkey]
            self.childStates[mkey] = self.state.playMove(mkey)
        
        self.edgePriors /= np.sum(self.edgePriors)

        self.reportCount = 0
        self.isExpanded = 1

    def exportNode(self, forNetwork, cnt=[0], path = None):
        """
        returns a structure that represents the tree.
        node: {
            state: str(state),
            edgeVisists: number[],
            edgeValues: number[],
            edgePriors: number[],
            virtualLosses: number,
            reportCount: number,
            children: node[]
        }

        only will include nodes that are expanded with the given network.
        """

        myPath = path
        if myPath is None:
            myPath = [self.state]

        if len(set(myPath)) < len(myPath):
            print("Cycle detected!")
            for mp in myPath:
                print(str(mp))
            print("======")
            assert False

        cnt[0] += 1
        if cnt[0] % 100000 == 0:
            print("Exporting node", cnt[0])

        if not self.isExpanded or self.networkId != forNetwork:
            return None

        # limit the size of the output by only exporting up to a certain depth
        if len(myPath) > 8:
            return None

        if np.sum(self.edgeTotalValues) == 0:
            return None

        result = dict()

        result["state"] = str(self.state)
        result["edgeVisits"] = self.edgeVisits.tolist()
        result["edgeValues"] = self.edgeTotalValues.tolist()
        result["virtualLosses"] = self.virtualLosses
        result["reportCount"] = self.reportCount

        children = []

        for child in dict.values(self.getChildren()):
            children.append(child.exportNode(forNetwork, cnt, myPath + [child.state]))

        children = list(filter(lambda x: x is not None, children))

        result["children"] = children

        return result

    def resetPriors(self, movePriors, generics, networkId, networkIteration, workerName):
        self.rawPriors = movePriors
        self.networkId = networkId
        self.networkIteration = networkIteration
        self.generics = generics
        self.workerName = workerName
        self.reportCount = 0

        # TODO so should these be reset or not?
        self.edgeVisits = np.zeros_like(self.legalMoveKeys, dtype=np.int32)
        self.edgeTotalValues = np.zeros_like(self.legalMoveKeys, dtype=np.float32)

        for i, mkey in enumerate(self.legalMoveKeys):
            self.edgePriors[i] = movePriors[mkey]
        self.edgePriors /= np.sum(self.edgePriors)

    def accessChild(self, moveKey):
        if moveKey in self.childNodes:
            return self.childNodes[moveKey]
        else:
            nextState = self.childStates[moveKey]
            if nextState in self.nodes:
                node = self.nodes[nextState]
                self.childNodes[moveKey] = node
                return node
            else:
                node = MCTSNode(nextState, self.nodes)
                self.nodes[nextState] = node
                self.childNodes[moveKey] = node
                return node

    def selectDown(self, cpuct, fpu, maxNodeAge, currentIteration):
        node = self
        passedNodes = []

        cdef int firstNode = 1

        while node.isExpanded and not node.state.hasEnded():

            if currentIteration - node.networkIteration >= maxNodeAge:
                break

            nextMove = node._pickMove(cpuct, fpu)

            if firstNode:
                firstNode = 0
            else:
                node.virtualLosses += 1

            passedNodes.append((node, nextMove))
            node = node.accessChild(node.legalMoveKeys[nextMove])

        failNode = node

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
        cdef int i
        for i in range(len(self.legalMoveKeys)):
            result[self.legalMoveKeys[i]] = self.edgeVisits[i]
        result /= float(np.sum(self.edgeVisits))
        return result

# standin for the policy ID of the first iteration
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
    def __init__(self, initialState, maxPackageSize, maxPendingPackages, evalAccess, cpuct, fpu, drawValue, networkIters, maxNodeAge, nodeRenewP):
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

        self.numUnreportedGamesPlayed = 0
        self.numPositionEvalsRequested = 0
        self.numUnreportedStates = 0

        self.networkIters = networkIters
        self.maxNodeAge = maxNodeAge
        self.nodeRenewP = nodeRenewP

        self.pendingReevals = set()

        self.pendingGames = 0

    def exportTree(self, forNetwork):
        logMsg("Beginning to exporting MCTS tree with %i nodes!" % len(self.root.nodes))
        exportedStructure = self.root.exportNode(forNetwork)
        logMsg("Completed export!")
        return json.dumps(exportedStructure)

    def backup(self, nodes, finalNode):
        """
        nodes is a list of non-terminal nodes which end with the given result.
        """

        result = finalNode.getTerminalResult()

        hadNewNode = False

        for node, move in nodes:
            node.backup(move, result, self.drawValue)
            if node.reportCount == 0:
                hadNewNode = True
            
        # the root node does not get virtual losses, they would not have an effect anyway. You have to visit it always.
        for node, move in nodes[1:]:
            node.removeVirtualLoss()

        mostRecentNetwork = None
        for network in self.networkIters:
            if self.networkIters[network] == len(self.networkIters):
                mostRecentNetwork = network
                break

        cdef int nidx

        if hadNewNode:
            for nidx, (node, move) in enumerate(nodes):
                assert node.isExpanded, "Only expanded nodes should be reported?!"

                iPolicy = (node.rawPriors, node.generics)
                policyUUID = node.networkId

                if mostRecentNetwork is None or policyUUID == mostRecentNetwork or node.reportCount == 0:
                    nextPolicy = None
                    if (nidx + 1) < len(nodes):
                        nextPolicy = nodes[nidx + 1][0].rawPriors
                    
                    record = packageReport(node.state, iPolicy, [finalNode.state.getWinnerNumber()], nextPolicy, policyUUID)
                    node.reportCount += 1
                    self.pendingReports.append(record)
                else:
                    self.numUnreportedStates += 1

            self.pendingReports[len(self.pendingReports)-1]["final"] = True
        else:
            self.numUnreportedGamesPlayed += 1

    def printReportCountStats(self):
        sumReports = 0
        numUnreported = 0
        numReported = 0
        numNodes = 0

        seenNodes = set()

        # a node that is not expanded and not terminal, so there should be games waiting for it.
        activeNodes = 0

        def analyze(node):
            nonlocal sumReports
            nonlocal numReported
            nonlocal numUnreported
            nonlocal numNodes
            nonlocal seenNodes
            nonlocal activeNodes

            seenNodes.add(node.state)

            if node.isExpanded:
                numNodes += 1
                if node.reportCount == 0:
                    numUnreported += 1
                else:
                    sumReports += node.reportCount
                    numReported += 1
            elif not node.state.hasEnded():
                activeNodes += 1
                
            children = list(dict.values(node.getChildren()))
            for child in children:
                if not (child.state in seenNodes):
                    analyze(child)
        
        analyze(self.root)

        print("\nThere are %i nodes with %i reports\n %i nodes have not made any report yet.\nActive nodes: %i\n" %\
            (numNodes, sumReports, numUnreported, activeNodes))

    def pollReports(self):
        ret = self.pendingReports
        self.pendingReports = []
        return ret
        
    def handleSelection(self, passedNodes, failNode):
        if failNode.state.hasEnded():
            self.backup(passedNodes, failNode)
        elif failNode.state in self.pendingEvalsInverse:
            evalId = self.pendingEvalsInverse[failNode.state]
            pendEval = self.pendingEvals[evalId]
            self.pendingEvals[evalId] = (pendEval[0], pendEval[1] + [passedNodes])
            self.pendingGames += 1
        elif failNode.state in self.newPendings:
            self.newPendings[failNode.state] = (failNode, self.newPendings[failNode.state][1] + [passedNodes])
            self.pendingGames += 1
        else:
            self.newPendings[failNode.state] = (failNode, [passedNodes])
            self.pendingGames += 1

    def incPackageSizes(self):
        self.currentMaxPackageSize *= 2
        if self.currentMaxPackageSize > self.maxPackageSize:
            self.currentMaxPackageSize = self.maxPackageSize
            self.currentMaxPendingPackages *= 2
            if self.currentMaxPendingPackages > self.maxPendingPackages:
                self.currentMaxPendingPackages = self.maxPendingPackages

        #logMsg("Limits are now: %i package size, %i packages" % (self.currentMaxPackageSize, self.currentMaxPendingPackages))

    def queuePendings(self, nextPending):
        package = list(dict.keys(nextPending))

        assert len(package) <= self.currentMaxPackageSize

        #logMsg("Queue a new package of size %i" % len(package))

        packageId = self.evalAccess.requestEvaluation(package)
        self.pendingEvalIds.add(packageId)

        cdef int idx

        for idx, g in enumerate(package):
            evalId = (packageId, idx)
            self.pendingEvalsInverse[g] = evalId
            self.pendingEvals[evalId] = (nextPending[g][0], nextPending[g][1])
            # this basically moves pending games from newPendings to pendingEvals, so self.pendingGames does not change.

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
                    logMsg("Warning: Submitting a package with only %i examples in it, as there is nothing else to do anymore!" % len(self.newPendings))
                    self.queuePendings(self.newPendings)
                    self.newPendings = dict()
                # else:
                #     logMsg("Will wait to fill current package, has size %i, needs %i!" % (len(self.newPendings), self.currentMaxPackageSize))

            #logMsg("Now waiting for %i packages with %i games!" % (len(self.pendingEvalIds), self.countPendingGames()))

    def countPendingGames(self):
        return self.pendingGames + len(self.pendingReevals)

    def canSpawnMoreGames(self):
        return len(self.newPendings) < self.currentMaxPackageSize and self.countPendingGames() < self.currentMaxPackageSize * self.currentMaxPendingPackages

    def handleNodeRenews(self, passedNodes):
        if self.nodeRenewP > 0 and self.canSpawnMoreGames():
            currentIteration = len(self.networkIters)
            for node, move in passedNodes:
                if node.isExpanded and node.networkIteration < currentIteration and np.random.random() < self.nodeRenewP and not (node.state in self.newPendings) and not (node.state in self.pendingReevals):
                    self.pendingReevals.add(node.state)
                    #print("Query renew for a node! There are now %i pending reevaluations!" % len(self.pendingReevals), node.networkId, node.networkIteration, "\n"+str(node.state))
                    self.newPendings[node.state] = (node, [])

    def fillPendingWithNews(self):
        cdef int lastSize = -1
        cdef int failCnt = 0
        cdef int cnt = 0
        while self.canSpawnMoreGames():
            if lastSize == len(self.newPendings):
                failCnt += 1
            else:
                failCnt = 0
            if failCnt > 50:
                break
            lastSize = len(self.newPendings)
            passedNodes, failNode = self.root.selectDown(self.cpuct, self.fpu, self.maxNodeAge, len(self.networkIters))
            cnt += 1
            self.handleSelection(passedNodes, failNode)
            self.handleNodeRenews(passedNodes)

        return cnt

    def beginGames(self):
        cdef int maxNew = 100
        cdef int newCount = 0
        startTime = time.monotonic()
        while self.allowNewGames and len(self.pendingEvalIds) < self.currentMaxPendingPackages and self.countPendingGames() < self.currentMaxPackageSize * self.currentMaxPendingPackages:
            maxNew -= 1
            newGames = self.fillPendingWithNews()
            if newGames == 0:
                logMsg("Cannot start more games!")
                break
            newCount += newGames
            self.handleNewPendings()

            if maxNew < 0:
                break
            
        if newCount > 0:
            logMsg("%i new games have started in %.2fs, now pending: %i games" % (newCount, (time.monotonic() - startTime), self.countPendingGames()))

    def hasOpenGames(self):
        return self.countPendingGames() > 0

    def rejectEvalsForNetwork(self, evalResults, latestNetwork):
        cdef int rejectedPackages = 0
        cdef int acceptedEvals = 0
        for uuid in list(dict.keys(evalResults)):
            if not (uuid in self.pendingEvalIds):
                continue
            
            results, network, workerName = evalResults[uuid]

            if network != latestNetwork:
                self.pendingEvalIds.remove(uuid)
                del evalResults[uuid]

                package = []

                for idx, (policy, generics) in enumerate(results):
                    evalId = (uuid, idx)
                    node, paths = self.pendingEvals[evalId]
                    package.append(node.state)

                self.numPositionEvalsRequested += len(package)
                packageId = self.evalAccess.requestEvaluation(package)
                self.pendingEvalIds.add(packageId)

                logMsg("!!!!!!!!!! Got network %s but expected %s on package %s from worker %s. Rewriting to package %s" % (network, latestNetwork, uuid, workerName, packageId))

                for idx, (policy, generics) in enumerate(results):
                    oldEvalId = (uuid, idx)
                    newEvalId = (packageId, idx)
                    node, paths = self.pendingEvals[oldEvalId]

                    self.pendingEvals[newEvalId] = self.pendingEvals[oldEvalId]
                    self.pendingEvalsInverse[node.state] = newEvalId

                    del self.pendingEvals[oldEvalId]

                rejectedPackages += 1
            else:
                acceptedEvals += len(results)

        if acceptedEvals > 0:
            logMsg("Accepted evaluations of %i states" % acceptedEvals)

        return rejectedPackages

    def continueGames(self, evalResults):
        self.incPackageSizes()

        for uuid in evalResults:
            if not (uuid in self.pendingEvalIds):
                continue
            self.pendingEvalIds.remove(uuid)

            results, network, workerName = evalResults[uuid]

            self.numPositionEvalsRequested += len(results)

            networkIter = 0
            if network in self.networkIters:
                networkIter = self.networkIters[network]

            currentIteration = len(self.networkIters)

            pendingContinuations = []
            for idx, (policy, generics) in enumerate(results):
                evalId = (uuid, idx)
                node, paths = self.pendingEvals[evalId]

                if node.isExpanded:
                    #logMsg("Resetting a prior for an already expanded node!")
                    node.resetPriors(policy, generics, network, networkIter, workerName)
                else:
                    node.expand(policy, generics, network, networkIter, workerName)

                for path in paths:
                    pendingContinuations.append((node, path))
                
                self.pendingReevals.discard(node.state)
                
                self.pendingGames -= len(paths)

                del self.pendingEvals[evalId]
                del self.pendingEvalsInverse[node.state]

            preSize = len(self.newPendings)

            for node, path in pendingContinuations:
                passedNodes, failNode = node.selectDown(self.cpuct, self.fpu, self.maxNodeAge, currentIteration)
                self.handleSelection(path + passedNodes, failNode)
                self.handleNodeRenews(passedNodes)

            postSize = len(self.newPendings)

            #logMsg("Continue %i games from %i positions to %i positions" % (len(pendingContinuations), len(results), postSize - preSize))

            self.handleNewPendings()


# there should only be one single of these, ran on a controlled machine.
# evaluation workers need to be started elsewhere and communicate via the evaluation-manager server process (hidden behind evalAccess implementation)
# with the TreeSelfPlayWorker. In frametime eval mode that has to be local, too. => make the frametime eval code patch the evalAccess implementation used.
class TreeSelfPlayWorker(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initialState, maxPackageSize, maxPendingPackages, maxNodeAge, nodeRenewP, gameReporter, evalAccess, cpuct, fpu, drawValue):
        self.initialState = initialState
        self.maxPackageSize = maxPackageSize
        self.maxPendingPackages = maxPendingPackages
        self.maxNodeAge = maxNodeAge
        self.nodeRenewP = nodeRenewP
        self.gameReporter = gameReporter
        self.evalAccess = evalAccess
        self.cpuct = cpuct
        self.fpu = fpu
        self.drawValue = drawValue

        self.seenNetworks = set()
        self.latestNetwork = None
        self.networkIterations = dict()

        self.commandHost = sys.argv[sys.argv.index("--command")+1]
        self.run = sys.argv[sys.argv.index("--run")+1]
        self.secret = sys.argv[sys.argv.index("--secret")+1]

        # it never steops using this one tree. Nodes with evaluations from old networks have a certain probability to request a new evaluation in new iterations
        # additionally they force a new evaluation if they are older than some number of iteration.
        # Configure this via maxNodeAge and nodeRenewP
        self.currentTree = self.newSelfPlayTree()

    def newSelfPlayTree(self):
        return SelfPlayTree(self.initialState, self.maxPackageSize, self.maxPendingPackages,\
                    self.evalAccess, self.cpuct, self.fpu, self.drawValue, self.networkIterations, self.maxNodeAge, self.nodeRenewP)

    def main(self):
        while True:
            self.playBatch()

    def initSelfplay(self, runId):
        pass

    def playBatch(self):

        newReports = []

        startTime = time.monotonic_ns()

        while len(newReports) == 0:
            evalResults = dict()

            newIteration = False
            prevNetwork = None

            while len(evalResults) == 0:
                self.currentTree.beginGames()
                evalResults = self.evalAccess.pollEvaluationResults()

                for uuid in evalResults:
                    results, network, workerName = evalResults[uuid]
                    if network is not None and not (network in self.seenNetworks):
                        self.seenNetworks.add(network)
                        prevNetwork = self.latestNetwork
                        self.latestNetwork = network
                        self.networkIterations[network] = len(self.seenNetworks)
                        newIteration = True

                rejects = 0
                rejects += self.currentTree.rejectEvalsForNetwork(evalResults, self.latestNetwork)

                if rejects > 0:
                    logMsg("!!!!!!!!!!!!!!!!!!! %i packages were rejected due to outdated networks used!" % rejects)

            if newIteration:
                # exportedTree = self.currentTree.exportTree(prevNetwork)
                # with open("/ImbaKeks/export_mcts.json", "w") as f:
                #     f.write(exportedTree)

                logMsg("=========================================================================")
                logMsg("=========================================================================")
                logMsg("==================== Detected a new iteration: %i!=======================" % (len(self.seenNetworks) + 1))
                logMsg("=========================================================================")
                logMsg("=========================================================================")
                logMsg("Known networks: ", self.seenNetworks)
                logMsg("Active network: %s" % self.latestNetwork)
                # logMsg("Stats of the iteration tree")
                # self.currentTree.printReportCountStats()
                logMsg("Unreported games played: %i" % self.currentTree.numUnreportedGamesPlayed)
                logMsg("States not reported due to old network: %i" % self.currentTree.numUnreportedStates)
                logMsg("Positions evaluated for this iteration: %i" % self.currentTree.numPositionEvalsRequested)

                drep = dict()
                drep["evals"] = self.currentTree.numPositionEvalsRequested
                drep["iteration"] = len(self.seenNetworks) - 1
                postJson(self.commandHost + "/api/evalscnt/" + self.run, self.secret, drep)

                self.currentTree.numUnreportedGamesPlayed = 0
                self.currentTree.numPositionEvalsRequested = 0
                self.currentTree.numUnreportedStates = 0

            self.currentTree.continueGames(evalResults)

            newReports += self.currentTree.pollReports()

        self.gameReporter.reportGame(newReports)

        batchTime = time.monotonic_ns()
        dt = (batchTime - startTime) / 1000000.0

        logMsg("Completed a batch, yielded %i new reports in %.2fms" % (len(newReports), dt))

        return (dt / len(newReports)), None, len(newReports)
