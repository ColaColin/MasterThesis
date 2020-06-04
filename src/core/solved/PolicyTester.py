import abc
import gzip
from utils.prints import logMsg, setLoggingEnabled
import time
from core.solved.TestDatabaseGenerator import getBestScoreKeys
import random
import numpy as np

class BatchedPolicyPlayer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getMoves(self, batch):
        """
        Given a batch of [(state, history)], find the best move to play and the expected result in all of those states. Should be able to handle any batchsize!
        @return (move, gameResult)[]. move is the index of the move, game result is -1 for "player on turn will lose", 0 for draw, 1 for "player on turn will win"
        """

class ShuffleBatchedPolicyPlayer(BatchedPolicyPlayer, metaclass=abc.ABCMeta):
    def getMoves(self, batch):
        result = []
        for state, _ in batch:
            legals = state.getLegalMoves()
            assert len(legals) > 0
            random.shuffle(legals)
            result.append((legals[0], random.choice([-1, 0, 1])))
        return result

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


class SolverBatchedPolicyPlayer(BatchedPolicyPlayer, metaclass=abc.ABCMeta):

    def __init__(self, solver):
        """
        solver is expected to be a GameSolver implementation
        """
        self.solver = solver

    def getMoves(self, batch):
        result = []
        for state, history in batch:
            assert not state.hasEnded()
            scores = self.solver.getMoveScores(state, history)
            keys = getBestScoreKeys(scores)
            if len(keys) > 0:
                result.append((random.choice(keys), sign(scores[keys[0]])))
            else:
                result.append((state.getLegalMoves()[0], 0))
        return result

class PolicyIteratorPlayer(BatchedPolicyPlayer, metaclass=abc.ABCMeta):
    def __init__(self, policy, policyIterator, policyUpdater, moveDecider, batchSize, playerParams = dict()):
        self.policy = policy
        self.policyIterator = policyIterator
        self.policyUpdater = policyUpdater
        self.moveDecider = moveDecider
        self.batchSize = batchSize

        if len(playerParams) > 0:
            logMsg("Will test with player configuration:", playerParams)

        self.playerParams = dict()
        self.playerParams[1] = playerParams
        self.playerParams[2] = playerParams

    def getMoves(self, batch):
        if not self.policyUpdater is None:
            self.policy = self.policyUpdater.update(self.policy)

        result = []

        while len(batch) > 0:
            miniBatch = [state for (state, _) in batch[:self.batchSize]]
            iteratedPolicy = self.policyIterator.iteratePolicy(self.policy, miniBatch, noExploration=True, playerHyperparametersDict=self.playerParams)
            result += list(map(lambda x: (self.moveDecider.decideMove(x[0], x[1][0], x[1][1]), 0), zip(miniBatch, iteratedPolicy)))

            batch = batch[self.batchSize:]

        return result

class PolicyPlayer(BatchedPolicyPlayer, metaclass=abc.ABCMeta):

    def __init__(self, policy, policyUpdater, moveDecider):
        self.policy = policy
        self.policyUpdater = policyUpdater
        self.moveDecider = moveDecider
    
    def getMoves(self, batch):
        if not self.policyUpdater is None:
            self.policy = self.policyUpdater.update(self.policy)
        
        def mapResult(game, absoluteWinPredictions):
            amax = np.argmax(absoluteWinPredictions)
            if amax == 0:
                return 0

            relativeWinner = game.mapPlayerNumberToTurnRelative(amax)
            if relativeWinner == 0:
                return 1
            else:
                return -1

        gbatch = list(map(lambda b: b[0], batch))
        movePredictions, winPredictions = list(zip(*self.policy.forward(gbatch)))
        movesResult = list(map(lambda x: np.argmax(x), movePredictions))
        winsResult = list(map(lambda x: mapResult(x[0], x[1]), zip(gbatch, winPredictions)))

        return list(zip(movesResult, winsResult))

def loadTestDataset(fpath, initialState):
    states = []
    solutions = []

    with open(fpath, "br") as f:
        exampleBytes = f.read()
    unzipped = gzip.decompress(exampleBytes)

    readingMoves = True
    state = initialState
    history = []
    solution = []
    for b in unzipped:
        wasReadingMoves = readingMoves
        readingMoves = b < initialState.getMoveCount()

        if readingMoves and wasReadingMoves:
            state = state.playMove(b)
            history.append(b)
        elif readingMoves and not wasReadingMoves:
            states.append((state, history))
            solutions.append(solution)
            state = initialState.playMove(b)
            history = [b]
            solution = []
        else: # notReadingMoves, do not care what we read before
            solution.append(b - initialState.getMoveCount())

    states.append((state, history))
    solutions.append(solution)

    return states, solutions

class DatasetPolicyTester():
    """
    Old, use version 2 below instead.

    Given a dataset and a PlayPolicy evaluate how well the policy predicts the best moves to play.
    A score of 100% is perfect, implying the policy always picked a move that eventually achieves the best gameplay result.
    With the TestDatabaseGenerator achieving the result in the fewest moves possible is not required.
    """

    def __init__(self, batchedPolicyPlayer, datasetFile, initialGameState, mode, batchSize):
        self.batchedPolicyPlayer = batchedPolicyPlayer
        self.datasetFile = datasetFile
        self.initialGameState = initialGameState
        self.states = []
        self.histories = []
        self.solutions = []
        self.mode = mode
        self.batchSize = batchSize

        self.loadExamples()

    def loadExamples(self):
        st, sol = loadTestDataset(self.datasetFile, self.initialGameState)
        self.states = st
        self.solutions = sol

    def runTest(self):
        startEval = time.monotonic()
        logMsg("Testing on %i examples!" % len(self.states))

        examples = list(zip(self.states, self.solutions))
        proced = 0
        lastPrint = 0
        hits = 0.0
        while len(examples) > 0:
            miniBatch = examples[:self.batchSize]

            inputs = [s for (s, _) in miniBatch]
            sols = [s for (_, s) in miniBatch]

            moves, _ = list(zip(*self.batchedPolicyPlayer.getMoves(inputs)))
            for midx, move in enumerate(moves):
                if move in sols[midx]:
                    hits += 1.0

            examples = examples[self.batchSize:]
            proced += len(miniBatch)

            if proced - lastPrint > 5000 and self.mode == "shell":
                lastPrint = proced
                logMsg("%.2f%%" % (100.0 * (proced / len(self.states))))
            
        accuracy = 100.0 * (hits / len(self.states))

        logMsg("Test on %i examples completed in %.2f seconds, result: %.2f%%" % (len(self.states), time.monotonic() - startEval, accuracy))

        return accuracy

    def main(self):
        setLoggingEnabled(self.mode == "shell")
        return self.runTest()
        
def loadExamples2(protoGame, datasetFile):
    """
    Loads a list of examples. An example is a tuple of 4 elements:
    - GameState object
    - History of moves that were played to get to the state, i.e. a list of move indices.
    - List of move indices considered to be correct in the position
    - Final result of the game under perfect play: -1 loss for current player, 0 draw, 1 win for current player.
    """
    startLoad = time.monotonic()
    with open(datasetFile, "br") as f:
        exampleBytes = f.read()
    unzipped = gzip.decompress(exampleBytes)
    bcount = len(unzipped)

    resultList = []

    def readNextLine(offset):
        ix = 0
        while bcount > offset + ix and unzipped[offset + ix] != 10:
            ix += 1
        line = unzipped[offset : offset + ix]
        return line
    
    offset = 0
    while True:
        nline = readNextLine(offset)
        lnline = len(nline)
        if lnline == 0:
            break
        else:
            offset += lnline
            offset += 1 # skips the linebreak in the file
            
            [pos, optimals, result] = nline.split(b' ')

            state = protoGame
            history = []
            for move in pos:
                # the file uses ascii 1 (code 49) for move index 0
                m = move - 49
                state = state.playMove(m)
                history.append(m)
            
            solution = []
            for optimal in optimals:
                # the file starts counting moves at ascii 1 (code 49), but playMove starts at 0
                solution.append(optimal - 49)

            # the file uses ascii 1 (code 49) for a draw
            # one less is a loss, one more is a win
            lineResult = (state, history, solution, result[0] - 49)

            resultList.append(lineResult)

    endLoad = time.monotonic()

    logMsg("Finished loading %i examples in %.2f seconds." % (len(resultList), endLoad - startLoad))

    return resultList
        
class DatasetPolicyTester2():
    def __init__(self, playerUnderTest, datasetFile, initialGameState, preloaded = None):
        self.playerUnderTest = playerUnderTest
        self.initialGameState = initialGameState

        if preloaded is None:
            [self.states, self.histories, self.solutions, self.gresults] =\
                list(zip(*loadExamples2(self.initialGameState, datasetFile)))
        else:
            [self.states, self.histories, self.solutions, self.gresults] = list(zip(*preloaded))

    def main(self):
        startEval = time.monotonic()
        logMsg("Begin test")

        testInput = list(zip(self.states, self.histories))
        movePredictions, resultPredictions = list(zip(*self.playerUnderTest.getMoves(testInput)))

        correctMoves = sum(map(lambda x: 1 if x[0] in x[1] else 0, zip(movePredictions, self.solutions)))
        correctGameResults = sum(map(lambda x: 1 if x[0] == x[1] else 0, zip(resultPredictions, self.gresults)))

        moveAccuracy = 100.0 * (correctMoves / len(self.states))
        resultAccuracy = 100.0 * (correctGameResults / len(self.states))

        evalTime = time.monotonic() - startEval

        logMsg("Test on %i examples took %.2f seconds, perfect play accuracies: for moves %.2f%% , for result: %.2f%%" 
            % (len(self.states), evalTime, moveAccuracy, resultAccuracy))

        return moveAccuracy, resultAccuracy

