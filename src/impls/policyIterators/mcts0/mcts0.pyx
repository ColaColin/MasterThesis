# cython: profile=False

from core.policy.PolicyIterator import PolicyIterator

import abc

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX

from utils.prints import logMsg

import random

cdef int bestLegalValue(float [:] ar):
    cdef int n = ar.shape[0]
    
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

    # a dict: move -> MCTSNode
    cdef object children

    # scale of the noise to be used if this is a root node. Should be between 0 and 1
    cdef float noiseMix

    # does this node have a network evaluation already?
    cdef int isExpanded

    # a tuple (MCTSnode, compressed move idx)
    cdef object parentNode

    # maps move indices to indices in the edge-arrays. This means those arrays can only track legal moves and
    # thus be smaller than the maximum number of moves possible, which is helpful in games with many illegal moves.
    cdef unsigned short [:] compressedMoveToMove

    # number moves possible in this state. Pick a number m between 0 and numMoves - 1 to represent a compressedMove
    # use compressedMoveToMove[m] to get the index as used by the game state
    cdef int numMoves

    # network output for the legal move edges
    cdef float [:] edgePriors
    # visits of legal moves
    cdef float [:] edgeVisits
    # sum of all values collected below a certain edge
    cdef float [:] edgeTotalValues

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

    def __init__(self, state, parentNode = None, noiseMix = 0.25):
        self.state = state

        self.noiseMix = noiseMix

        self.isExpanded = 0

        self.parentNode = parentNode

        self.children = {}

        self.terminalResult = None

        self.noiseCache = None

        self.stateValue = 0

        self.allVisits = 0

        self.edgePriors = None

        # these are all only required once the first select move is to be called,
        # which on some (many?) leafs is never done, so they are initialized in a lazy fashion once needed
        self.numMoves = -1
        self.compressedMoveToMove = None
        self.edgeVisits = None
        self.edgeTotalValues = None

    def getState(self):
        return self.state

    cdef void _lazyInitEdgeData(self):
        if self.numMoves == -1:
            legalMoves = self.state.getLegalMoves()
            self.numMoves = len(legalMoves)
            assert self.state.getMoveCount() < 65536, "If you need more than 65k possible moves use uint32 below!"
            self.compressedMoveToMove = np.array(legalMoves, dtype=np.uint16)
            self.edgeVisits = np.zeros(self.numMoves, dtype=np.float32)
            self.edgeTotalValues = np.zeros(self.numMoves, dtype=np.float32)

    cdef int _pickMove(self, float cpuct):

        cdef int useNoise = self.parentNode == None and self.noiseMix > 0

        if useNoise and self.noiseCache is None:
            self.noiseCache = np.random.dirichlet(getDconst(self.numMoves)).astype(np.float32)

        cdef int i

        cdef float nodeQ, nodeU

        # .00001 means that in the case of a new node with zero visits it will chose whatever has the best P
        # instead of just the move with index 0 (for a constant FPU at least)
        # but there is little effect in other cases
        cdef float visitsRoot = self.allVisits ** 0.5 + 0.00001

        cdef float [:] valueTmp = np.zeros(self.numMoves, dtype=np.float32)

        cdef int decompressedMove

        # first play urgency. Value of moves that have not been considered yet. Setting it to 0 (=losing move)
        # is what the original AlphaZero implementation did. There are better ways to handle this value, tbd
        # Using 0 is pretty bad however, as it makes it hard to learn from a policy that knows
        # the correct winners, as they're never even looked at if there was a bad move probability estimation
        # That's why this uses 0.45 instead, so assume an unplay move is likely slightly worse than a draw
        # further improvements are probably possible here
        cdef float fpu = 0.45

        for i in range(self.numMoves):
            # TODO consider reducing the size of edgePriors to only contain legal moves
            decompressedMove = self.compressedMoveToMove[i]

            if useNoise:
                valueTmp[i] = (1 - self.noiseMix) * self.edgePriors[decompressedMove] + self.noiseMix * self.noiseCache[i]
            else:
                valueTmp[i] = self.edgePriors[decompressedMove]

            if self.edgeVisits[i] == 0:
                nodeQ = fpu
            else:
                nodeQ = self.edgeTotalValues[i] / self.edgeVisits[i]

            nodeU = valueTmp[i] * (visitsRoot / (1.0 + self.edgeVisits[i]))

            valueTmp[i] = nodeQ + cpuct * nodeU

        cdef int result = bestLegalValue(valueTmp)

        return self.compressedMoveToMove[result]

    cdef MCTSNode _executeMove(self, int move):
        cdef object newState = self.state.playMove(move)

        cdef MCTSNode knownNode

        cdef int ix
        cdef int compressedNodeIdx = -1

        for ix in range(self.numMoves):
            if self.compressedMoveToMove[ix] == move:
                compressedNodeIdx = ix
                break

        cdef MCTSNode newNode = MCTSNode(newState, (self, compressedNodeIdx), self.noiseMix)

        return newNode

    cdef MCTSNode _selectMove(self, float cpuct):
        self._lazyInitEdgeData()

        cdef int move = self._pickMove(cpuct)

        if not move in self.children:
            self.children[move] = self._executeMove(move)
        
        return self.children[move]

    cdef void backup(self, object vs, float drawValue):
        """
        backup results found in a leaf up the tree
        @param vs: win chances by the network, indexed by player numbers, 0 stands for draw chance.
        """

        if self.parentNode == None:
            return

        cdef MCTSNode pNode = self.parentNode[0]
        cdef int pMove = self.parentNode[1]
        
        pNode.edgeVisits[pMove] += 1
        pNode.allVisits += 1
        pNode.edgeTotalValues[pMove] += vs[pNode.state.getPlayerOnTurnNumber()] + vs[0] * drawValue
        pNode.backup(vs, drawValue)

    
    cdef void expand(self, object movePMap, object vs, float drawValue):
        """
        Fill in missing network evaluations, allowing to select a move on this node
        @param movePMap: move policy of the network
        @param vs: win chances by the network, indexed by player numbers, 0 stands for draw chance.
        """
        self.edgePriors = np.zeros(self.state.getMoveCount(), dtype=np.float32)
        np.copyto(np.asarray(self.edgePriors), movePMap, casting="no")

        self.isExpanded = 1
        self.netValueEvaluation = vs
        self.stateValue = vs[self.state.getPlayerOnTurnNumber()] + vs[0] * drawValue

    cdef MCTSNode selectDown(self, float cpuct):
        """
        return a leaf that was chosen by selecting good moves going down the tree
        """
        cdef MCTSNode node = self
        while node.isExpanded and not node.state.hasEnded():
            node = node._selectMove(cpuct)
        return node

    cdef object getTerminalResult(self):
        """
        Gets the target value to learn for a network that wants to predict the outcome of this terminal state.
        Do not call unless state.hasEnded() is true!
        """
        if self.terminalResult is None:
            assert self.state.hasEnded()
            numOutputs = self.state.getPlayerCount() + 1
            r = [0] * numOutputs
            winner = self.state.getWinnerNumber()
            r[winner] = 1
            self.terminalResult = np.array(r, dtype=np.float32)

        return self.terminalResult

    cdef object getMoveDistribution(self):
        """
        The distribution over all moves that represents a policy likely better than the one provided by the network alone.
        """
        result = np.zeros(self.state.getMoveCount(), dtype=np.float32)
        result[self.compressedMoveToMove] = self.edgeVisits
        result /= float(self.allVisits)

        return result

class MctsPolicyIterator(PolicyIterator, metaclass=abc.ABCMeta):
    """
    AlphaZero-style MCTS implementation extended with explicit handling of draws.
    FPU for now is just 0, following AlphaZero basics
    """

    def __init__(self, expansions, cpuct, rootNoise, drawValue):
        logMsg("Creating MctsPolicyIterator(expansions=%i, cpuct=%f,rootNoise=%f, drawValue=%f)" % (expansions, cpuct, rootNoise, drawValue))
        self.expansions = expansions
        self.cpuct = cpuct
        self.rootNoise = rootNoise
        self.drawValue = drawValue
  
    def backupWork(self, backupSet, evalout):
        cdef MCTSNode node
        cdef int idx

        for idx, ev in enumerate(evalout):
            node = backupSet[idx]
            w = ev[1]
            if node.state.hasEnded():
                w = node.getTerminalResult()
            else:
                node.expand(ev[0], ev[1], self.drawValue)
            
            node.backup(w, self.drawValue)

    def cpuWork(self, prepareSet, backupSet, evalout):
        prepareResult = []
        
        cdef MCTSNode tnode
        
        if backupSet is not None:
            self.backupWork(backupSet, evalout)
        
        for i in range(len(prepareSet)):
            tnode = prepareSet[i]
            prepareResult.append(tnode.selectDown(self.cpuct))

        return prepareResult

    def iteratePolicy(self, policy, gamesBatch):
        nodes = [MCTSNode(g, noiseMix = self.rootNoise) for g in gamesBatch]

        halfw = len(nodes) // 2

        nodesA = nodes[:halfw]
        nodesB = nodes[halfw:]

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

        preparedDataA = self.cpuWork(nodesA, None, None)
        evaloutA = policy.forward([p.getState() for p in preparedDataA])

        preparedDataB = self.cpuWork(nodesB, None, None)
        evaloutB = None

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
        
        cdef int e, ex
        for e in range(self.expansions):
            for ex in range(2):
                if asyncA:
                    evaloutB = policy.forward([p.getState() for p in preparedDataB], asyncCall = asyncWork)
                else:
                    evaloutA = policy.forward([p.getState() for p in preparedDataA], asyncCall = asyncWork)
                
                asyncA = not asyncA

        self.backupWork(preparedDataA, evaloutA)

        result = []

        cdef MCTSNode node

        for node in nodes:
            generics = dict()
            generics["net_values"] = [x for x in node.netValueEvaluation]
            result.append((node.getMoveDistribution(), generics))

        return result
