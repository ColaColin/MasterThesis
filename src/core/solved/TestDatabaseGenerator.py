import time
from utils.prints import logMsg
import abc
import io
import gzip

class TestPlayGeneratorPolicy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decideMoves(self, state, solverMoveScores, bestMoveKeys):
        """
        Play at least one move, a list of multiple moves is also possible,
        to branch out the tree of games more. 
        """

def getBestScoreKeys(scoreDict):
    bestScore = None
    for k in scoreDict:
        if bestScore is None or scoreDict[k] > bestScore:
            bestScore = scoreDict[k]
    
    results = []
    for k in scoreDict:
        if scoreDict[k] == bestScore:
            results.append(k)
    return results

class TestDatabaseGenerator():

    """
    Play out a tree of game states according to some policy, which also has access
    to the solvers solution.
    Use a solver to store which moves do not make the situation worse.
    Finding shortest wins is not required, but winning when winning is possible
    and drawing when drawing is possible is.
    The database is stored as a byte array:
    a list of values from 0-(n-1), representing the moves played to reach the position
    correct moves (e.g. achieve optimal result eventually) listed as values from n-(2n-1)
    then again values from 0-(n-1), signaling the next game.
    """

    def __init__(self, initialState, solver, databaseSize, outputFile, policy, generation):
        self.initialState = initialState
        self.solver = solver
        self.databaseSize = databaseSize
        self.outputFile = outputFile
        self.results = [] # pairs of game state, list of correct moves
        self.policy = policy
        self.generation = generation

    def confGeneration(self, movePlacer):
        resultDict = dict()
        iter = 0
        while self.databaseSize > len(resultDict):
            iter += 1    
            nextVisits = [(self.initialState, [])]

            while len(nextVisits) > 0 and self.databaseSize > len(resultDict):
                state, path = nextVisits.pop(0)
                solution = self.solver.getMoveScores(state, path)
                optimals = getBestScoreKeys(solution)
                if not state in resultDict and len(state.getLegalMoves()) > len(optimals):
                    resultDict[state] = (path, optimals)
                    #print(state, path, state.getLegalMoves(), solution, optimals)
                #print("----")

                if len(resultDict) % 500 == 0:
                    logMsg("Tree %i" % iter, str(100 * ((len(resultDict)) / self.databaseSize)) + "%")
                moves = self.policy.decideMoves(state, solution, optimals)
                for i, move in enumerate(moves):
                    nextState = state.playMove(move)
                    if not nextState.hasEnded():
                        addFront = movePlacer(i)
                        obj = (nextState, path + [move])
                        if addFront:
                            nextVisits.insert(0, obj)
                        else:
                            nextVisits.append(obj)

        for s in resultDict:
            self.results.append(resultDict[s])

    def breadthFirstGeneration(self):
        self.confGeneration(lambda i: False)

    def beamGeneration(self):
        self.confGeneration(lambda i: i == 0)

    def package(self):
        buffer = io.BytesIO()
        for path, optimals in self.results:
            buffer.write(bytes([p for p in path]))
            buffer.write(bytes([o + self.initialState.getMoveCount() for o in optimals]))

        bys = buffer.getvalue()
        zipped = gzip.compress(bys)
        logMsg("Package size is %i kb, compressed to %i kb" % ((len(bys) // 1024), (len(zipped) // 1024)))
        with open(self.outputFile, "wb") as f:
            f.write(zipped)

    def main(self):
        generationStart = time.monotonic()

        if self.generation == "beam":
            self.beamGeneration()
        else:
            self.breadthFirstGeneration()

        generationEnd = time.monotonic()

        logMsg("Generated %i positions within %i seconds." % (len(self.results), int(generationEnd-generationStart)))

        depthCounts = dict()
        for p, _ in self.results:
            k = len(p)
            if not k in depthCounts:
                depthCounts[k] = 0
            depthCounts[k] += 1
        
        logMsg("Depth distribution is:", depthCounts)

        self.package()



