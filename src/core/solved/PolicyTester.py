import abc
import gzip
from utils.prints import logMsg, setLoggingEnabled
import time
from core.solved.TestDatabaseGenerator import getBestScoreKeys
import random

class BatchedPolicyPlayer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getMoves(self, batch):
        """
        Given a batch of [(state, history)], find the best move to play in all of those states. Should be able to handle any batchsize!
        """

class ShuffleBatchedPolicyPlayer(BatchedPolicyPlayer, metaclass=abc.ABCMeta):
    def getMoves(self, batch):
        result = []
        for state, _ in batch:
            legals = state.getLegalMoves()
            assert len(legals) > 0
            random.shuffle(legals)
            result.append(legals[0])
        return result

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
                result.append(keys[0])
            else:
                result.append(state.getLegalMoves()[0])
        return result

class PolicyIteratorPlayer(BatchedPolicyPlayer, metaclass=abc.ABCMeta):
    def __init__(self, policy, policyIterator, policyUpdater, moveDecider, batchSize, quickFactor = 1):
        self.policy = policy
        self.policyIterator = policyIterator
        self.policyUpdater = policyUpdater
        self.moveDecider = moveDecider
        self.batchSize = batchSize
        self.quickFactor = quickFactor

    def getMoves(self, batch):
        if not self.policyUpdater is None:
            self.policy = self.policyUpdater.update(self.policy)

        result = []

        while len(batch) > 0:
            miniBatch = [state for (state, _) in batch[:self.batchSize]]
            iteratedPolicy = self.policyIterator.iteratePolicy(self.policy, miniBatch, noExploration=True, quickFactor=self.quickFactor)
            result += list(map(lambda x: self.moveDecider.decideMove(x[0], x[1][0], x[1][1]), zip(miniBatch, iteratedPolicy)))

            batch = batch[self.batchSize:]

        return result

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

            moves = self.batchedPolicyPlayer.getMoves(inputs)
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
        
        

