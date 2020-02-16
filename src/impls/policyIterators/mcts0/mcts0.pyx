# cython: profile=False

from core.base.PolicyIterator import PolicyIterator

import abc

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX

from utils.prints import logMsg

from libc.stdlib cimport free

from utils.fields.fields cimport mallocWithZero

import random

cdef int bestLegalValue(float* ar, int n):
    if n == 0:
        return -1
    
    # randomize where to start the iteration over the candidate moves
    # such that ties between moves are not resolved by a "low index moves win"-policy,
    # preventing a bias to play low index moves.
    cdef int startIdx = int((rand()/(<float>RAND_MAX)) * n)
    
    cdef int idx, bestIdx, i
    cdef float bestValue
    
    bestIdx = -1
    bestValue = -1
    
    for i in range(n):
        idx = (i + startIdx) % n
        
        if ar[idx] > bestValue:
            bestValue = ar[idx]
            bestIdx = idx
    
    return bestIdx

cdef object dconsts = {}

cdef object getDconst(int n):
    global dconsts
    if not n in dconsts:
        dconsts[n] = np.asarray([10.0 / n] * n, dtype=np.float32)
    return dconsts[n]

cdef class MCTSNode():
    # a GameState object
    cdef readonly object state

    # cached playerOnTurnNumber for the state state. This is used a lot and thus it would be expensive to call the abstract state object for it a lot
    cdef int playerOnTurnNumber

    # cached hasEnded for the state
    cdef int hasEnded

    # a dict: move -> MCTSNode
    cdef object children

    # scale of the noise to be used if this is a root node. Should be between 0 and 1
    cdef float noiseMix

    # does this node have a network evaluation already?
    cdef int isExpanded

    # the parent node from which this node was reached. Is None for a root node
    cdef MCTSNode parentNode
    # the move key that was played in the parent to reach this node
    cdef int parentMove

    # number moves possible in this state. Pick a number m between 0 and numMoves - 1 to represent a legal move
    # use legalMoveKeys[m] to get the index as used by the game state
    cdef int numMoves

    # data of edgePriors, edgeVisits, edgeTotalValues. Not noisecache, as for most nodes it is null anyway.
    # All allocated in a single call for more speed.
    # the pointers below just point at this data at different offsets.
    cdef float* edgeData

    # network output for the legal move edges
    cdef float* edgePriors
    # visits of legal moves
    cdef float* edgeVisits
    # sum of all values collected below a certain edge
    cdef float* edgeTotalValues

    # keys of legal moves that the edges above refer to
    cdef int* legalMoveKeys

    # cache of the noise used if this is a root node
    cdef float[:] noiseCache

    # if the state is terminal this is the result
    cdef object terminalResult

    # the value of the state according to the network
    cdef float stateValue

    # how many times was this node visited
    cdef int allVisits

    # the raw network output for the state value
    cdef object netValueEvaluation

    def __init__(self, state, MCTSNode parentNode = None, int parentMove = -1, float noiseMix = 0.25):
        self.state = state

        self.playerOnTurnNumber = self.state.getPlayerOnTurnNumber()

        self.hasEnded = self.state.hasEnded()

        self.noiseMix = noiseMix

        self.isExpanded = 0

        self.parentNode = parentNode
        self.parentMove = parentMove

        self.children = {}

        self.terminalResult = None

        self.noiseCache = None

        self.stateValue = 0

        self.allVisits = 0

        self.numMoves = -1

    def __dealloc__(self):
        if self.numMoves != -1:
            free(self.edgeData)
            free(self.legalMoveKeys)

    def getState(self):
        return self.state

    cdef int _pickMove(self, float cpuct, float fpu):

        cdef int useNoise = self.parentNode is None and self.noiseMix > 0

        if useNoise and self.noiseCache is None:
            self.noiseCache = np.random.dirichlet(getDconst(self.numMoves)).astype(np.float32)

        cdef int i

        cdef float nodeQ, nodeU

        # .00001 means that in the case of a new node with zero visits it will chose whatever has the best P
        # instead of just the move with index 0 (for a constant FPU at least)
        # but there is little effect in other cases
        cdef float visitsRoot = self.allVisits ** 0.5 + 0.00001

        # this will still contain old numbers, but those are overwritten anyway below!
        cdef float* valueTmp = &self.edgeData[self.numMoves * 3]

        for i in range(self.numMoves):
            if useNoise:
                valueTmp[i] = (1 - self.noiseMix) * self.edgePriors[i] + self.noiseMix * self.noiseCache[i]
            else:
                valueTmp[i] = self.edgePriors[i]

            if self.edgeVisits[i] == 0:
                nodeQ = fpu
            else:
                nodeQ = self.edgeTotalValues[i] / self.edgeVisits[i]

            nodeU = valueTmp[i] * (visitsRoot / (1.0 + self.edgeVisits[i]))

            valueTmp[i] = nodeQ + cpuct * nodeU

        return bestLegalValue(valueTmp, self.numMoves)

    cdef MCTSNode _executeMove(self, int moveIndex):
        cdef object newState = self.state.playMove(self.legalMoveKeys[moveIndex])

        cdef MCTSNode knownNode

        cdef MCTSNode newNode = MCTSNode(newState, self, moveIndex, self.noiseMix)

        return newNode

    cdef MCTSNode _selectMove(self, float cpuct, float fpu):
        cdef int moveIndex = self._pickMove(cpuct, fpu)

        if not moveIndex in self.children:
            self.children[moveIndex] = self._executeMove(moveIndex)
        
        return self.children[moveIndex]

    cdef void backup(self, float[:] vs, float drawValue):
        """
        backup results found in a leaf up the tree
        @param vs: win chances by the network, indexed by player numbers, 0 stands for draw chance.
        """

        cdef MCTSNode parentNode = self.parentNode

        if parentNode is None:
            return

        parentNode.edgeVisits[self.parentMove] += 1
        parentNode.allVisits += 1
        parentNode.edgeTotalValues[self.parentMove] += vs[parentNode.playerOnTurnNumber] + vs[0] * drawValue
        parentNode.backup(vs, drawValue)

    
    cdef void expand(self, float[:] movePMap, float[:] vs, float drawValue):
        """
        Fill in missing network evaluations, allowing to select a move on this node
        @param movePMap: move policy of the network
        @param vs: win chances by the network, indexed by player numbers, 0 stands for draw chance.
        """

        legalMoves = self.state.getLegalMoves()
        self.numMoves = len(legalMoves)

        # the 4th entry is used as temporary storage when running _pickMove
        self.edgeData = <float*> mallocWithZero(4 * self.numMoves * sizeof(float))

        self.edgePriors = self.edgeData
        self.edgeVisits = &self.edgeData[self.numMoves]
        self.edgeTotalValues = &self.edgeData[self.numMoves*2]

        self.legalMoveKeys = <int*> mallocWithZero(self.numMoves * sizeof(int))

        cdef int i, mv
        for i in range(self.numMoves):
            mv = legalMoves[i]
            self.legalMoveKeys[i] = mv
            self.edgePriors[i] = movePMap[mv]

        self.isExpanded = 1
        self.netValueEvaluation = vs
        self.stateValue = vs[self.playerOnTurnNumber] + vs[0] * drawValue

    cdef MCTSNode selectDown(self, float cpuct, float fpu):
        """
        return a leaf that was chosen by selecting good moves going down the tree
        """
        cdef MCTSNode node = self
        while node.isExpanded and not node.hasEnded:
            node = node._selectMove(cpuct, fpu)
        return node

    cdef object getTerminalResult(self):
        """
        Gets the target value to learn for a network that wants to predict the outcome of this terminal state.
        Do not call unless state.hasEnded() is true!
        """
        if self.terminalResult is None:
            assert self.hasEnded
            numOutputs = self.state.getPlayerCount() + 1
            r = [0] * numOutputs
            winner = self.state.getWinnerNumber()
            r[winner] = 1
            self.terminalResult = np.array(r, dtype=np.float32)

        return self.terminalResult

    cdef object getNetworkPriors(self):
        """
        Get the network prior edge outputs for legal moves. Likely do not sum to 1.
        Meant mainly for better understanding of the learning, not required by the algorithm.
        """
        result = np.zeros(self.state.getMoveCount(), dtype=np.float32)
        cdef int i
        for i in range(self.numMoves):
            result[self.legalMoveKeys[i]] = self.edgePriors[i]

        return result

    cdef object getMoveDistribution(self):
        """
        The distribution over all moves that represents a policy likely better than the one provided by the network alone.
        """
        result = np.zeros(self.state.getMoveCount(), dtype=np.float32)
        cdef int i
        for i in range(self.numMoves):
            result[self.legalMoveKeys[i]] = self.edgeVisits[i]
        result /= float(self.allVisits)

        return result

class MctsPolicyIterator(PolicyIterator, metaclass=abc.ABCMeta):
    """
    AlphaZero-style MCTS implementation extended with explicit handling of draws.
    FPU can be configured, AlphaZero standard is 0. There must be better ways to handle it, however.
    """

    def __init__(self, expansions, cpuct, rootNoise, drawValue, fpu = 0.45):
        logMsg("Creating MctsPolicyIterator(expansions=%i, cpuct=%f,rootNoise=%f, drawValue=%f,fpu=%f)" % (expansions, cpuct, rootNoise, drawValue, fpu))
        self.expansions = expansions
        self.cpuct = cpuct
        self.rootNoise = rootNoise
        self.drawValue = drawValue
        self.fpu = fpu
  
    def backupWork(self, list backupSet, list evalout):
        cdef MCTSNode node
        cdef int idx

        for idx, ev in enumerate(evalout):
            node = backupSet[idx]
            w = ev[1]
            if node.hasEnded:
                w = node.getTerminalResult()
            else:
                node.expand(ev[0], ev[1], self.drawValue)
            
            node.backup(w, self.drawValue)

    def cpuWork(self, list prepareSet, list backupSet, list evalout):
        cdef list prepareResult = []
        
        cdef MCTSNode tnode
        
        if backupSet is not None:
            self.backupWork(backupSet, evalout)
        
        for i in range(len(prepareSet)):
            tnode = prepareSet[i]
            prepareResult.append(tnode.selectDown(self.cpuct, self.fpu))

        return prepareResult

    def iteratePolicy(self, policy, gamesBatch, noExploration = False, quickFactor = 1):
        noise = 0 if noExploration else self.rootNoise
        cdef list nodes = [MCTSNode(g, noiseMix = noise) for g in gamesBatch]

        cdef int halfw = len(nodes) // 2

        cdef list nodesA = nodes[:halfw]
        cdef list nodesB = nodes[halfw:]

        # the pipeline goes:

        # prepare A
        # evaluate A
        # prepare B
        
        # GPU        - CPU
        # do:
        # evaluate B - backup A, prepare A 
        # evaluate A - backup B, prepare B
        # repeat
        
        # complete
        # backupA

        asyncA = True

        cdef list preparedDataA = self.cpuWork(nodesA, None, None)
        cdef list evaloutA = policy.forward([p.getState() for p in preparedDataA])

        cdef list preparedDataB = self.cpuWork(nodesB, None, None)
        cdef list evaloutB = None

        me = self

        def asyncWork():
            nonlocal preparedDataA
            nonlocal preparedDataB
            nonlocal asyncA
            nonlocal me
            nonlocal evaloutA
            nonlocal evaloutB
            nonlocal nodesA
            nonlocal nodesB

            if asyncA:
                preparedDataA = me.cpuWork(nodesA, preparedDataA, evaloutA)
            else:
                preparedDataB = me.cpuWork(nodesB, preparedDataB, evaloutB)
        
        cdef int nodeExpansions
        nodeExpansions = self.expansions // quickFactor
        if nodeExpansions < 1:
            nodeExpansions = 1

        cdef int e, ex
        for e in range(nodeExpansions):
            for ex in range(2):
                if asyncA:
                    evaloutB = policy.forward([p.getState() for p in preparedDataB], asyncCall = asyncWork)
                else:
                    evaloutA = policy.forward([p.getState() for p in preparedDataA], asyncCall = asyncWork)
                
                asyncA = not asyncA

        self.backupWork(preparedDataA, evaloutA)

        cdef list result = []

        cdef MCTSNode node

        for node in nodes:
            generics = dict()
            generics["net_values"] = [x for x in node.netValueEvaluation]
            generics["net_priors"] = node.getNetworkPriors()
            result.append((node.getMoveDistribution(), generics))

        return result
