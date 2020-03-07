import time
from utils.prints import logMsg
import abc
import io
import gzip

from multiprocessing import Process, Queue

from utils.misc import constructor_for_class_name

import os

import random

import numpy as np

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
    Old class, use v2 further below this
    The database is stored as a byte array:
    a list of values from 0-(n-1), representing the moves played to reach the position
    correct moves (e.g. achieve optimal result eventually) listed as values from n-(2n-1)
    then again values from 0-(n-1), signaling the next game.

    There are two modes:
    - strong: Moves are only considered correct if they lead to the fastest win or prolong losing the longest possible.
    - weak: Moves are considered correct if they lead to any possible win or prolong losing the longest possible.
    """

    def __init__(self, initialState, solver, databaseSize, outputFile, policy, generation, mode = "stong", dedupe=True, filterSimple=True):
        self.initialState = initialState
        self.solver = solver
        self.databaseSize = databaseSize
        self.outputFile = outputFile
        self.results = [] # pairs of game state, list of correct moves
        self.policy = policy
        self.generation = generation
        self.mode = mode
        self.dedupe = dedupe
        self.filterSimple = filterSimple

        if self.mode != "strong" and self.mode != "weak":
            raise Exception("Unknown mode: " + self.mode)

    def weakenSolution(self, solution):
        result = dict()
        for m in solution:
            s = solution[m]
            if s > 0:
                result[m] = 1
            else:
                result[m] = s 
        return result

    def confGeneration(self, movePlacer):
        resultSet = set()

        iter = 0
        while self.databaseSize > len(self.results):
            iter += 1    
            nextVisits = [(self.initialState, [])]

            while len(nextVisits) > 0 and self.databaseSize > len(self.results):
                state, path = nextVisits.pop(0)
                presol = self.solver.getMoveScores(state, path)
                
                if self.mode == "weak":
                    solution = self.weakenSolution(presol)
                else:
                    solution = presol

                optimals = getBestScoreKeys(solution)

                if not state in resultSet:
                    isSimple = len(state.getLegalMoves()) == len(optimals)
                    if (not isSimple) or (not self.filterSimple):
                        self.results.append((path, optimals))
                    if self.dedupe:
                        resultSet.add(state)

                    #print(state, path, state.getLegalMoves(), presol, solution, optimals)
                #print("----")

                if len(self.results) % 500 == 0:
                    logMsg("Tree %i" % iter, str(100 * ((len(self.results)) / self.databaseSize)) + "%")

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
        optMoveCounts = 0
        for p, optimals in self.results:
            k = len(p)
            if not k in depthCounts:
                depthCounts[k] = 0
            depthCounts[k] += 1
            optMoveCounts += len(optimals)
        
        logMsg("Depth distribution is:", depthCounts)
        logMsg("Average number of correct moves is:", optMoveCounts / len(self.results))

        self.package()


def getGameResult(solution):
    bestResult = -1
    for score in dict.values(solution):
        if bestResult < score:
            bestResult = score
    if bestResult > 1:
        bestResult = 1
    return bestResult

def playGames(outputQueue, gameCtor, gameCtorParams, solver, policy, dedupe, filterTrivial):
    initialState = constructor_for_class_name(gameCtor)(**gameCtorParams)

    solveCache = dict()

    results = []

    while True:
        state = initialState
        path = []

        while not state.hasEnded():
            wasMiss = False
            if not state in solveCache:
                solution = solver.getMoveScores(state, path)
                solveCache[state] = solution
                wasMiss = True
            else:
                solution = solveCache[state]

            optimals = getBestScoreKeys(solution)
            isSimple = len(state.getLegalMoves()) == len(optimals)

            if not isSimple or not filterTrivial:
                if not dedupe or wasMiss:
                    results.append((state.store(), path.copy(), optimals, getGameResult(solution), ))

                if len(results) > 100:
                    outputQueue.put(results)
                    results = []

            moves = policy.decideMoves(state, solution, optimals)
            move = random.choice(moves)
            
            state = state.playMove(move)
            path.append(move)

class TestDatabaseGenerator2():

    def __init__(self, initialState, solver, databaseSize, outputFile, policy, dedupe, workers, filterTrivial):
        self.initialState = initialState
        self.solver = solver
        self.databaseSize = databaseSize
        self.outputFile = outputFile
        self.policy = policy
        self.dedupe = dedupe
        self.filterTrivial = filterTrivial
        self.workers = workers
        self.workerprocs = []

    def main(self, timeout = None):
        statesQueue = Queue(maxsize=50)
        if timeout is None:
            timeout = 9999999999999
        deadline = time.monotonic() + timeout

        for _ in range(self.workers):
            proc = Process(target=playGames, args=(statesQueue, self.initialState.getGameConstructorName(), self.initialState.getGameConstructorParams(), self.solver, self.policy, self.dedupe, self.filterTrivial))
            proc.daemon = True
            proc.start()
            self.workerprocs.append(proc)

        results = []
        resultsSet = set()
        simpleCount = 0

        waitTimes = []

        startFrame = time.monotonic_ns()
        while len(results) < self.databaseSize:

            if time.monotonic() > deadline:
                raise Exception("Timeout generating dataset!")

            try:
                resultList = statesQueue.get(timeout=8)
            except:
                continue

            for stateStore, path, optimals, gameResult in resultList:
                state = self.initialState.load(stateStore)

                if not self.dedupe or (not state in resultsSet):
                    if len(results) < self.databaseSize:
                        results.append((path, optimals, gameResult))
                    else:
                        break

                    frameTime = time.monotonic_ns() - startFrame
                    startFrame = time.monotonic_ns()
                    waitTimes.append(frameTime)

                    if len(waitTimes) > 3000:
                        del waitTimes[0]

                    if len(results) % 1000 == 0:
                        meanWait = np.mean(waitTimes)
                        logMsg("Generated %i examples, current speed %.2f per second." % (len(results), 1000000000 / meanWait))

                if self.dedupe:
                    resultsSet.add(state)

        depthCounts = dict()
        optMoveCounts = 0
        winMoves = 0
        lossMoves = 0
        drawMoves = 0
        for p, optimals, gresult in results:
            k = len(p)
            if not k in depthCounts:
                depthCounts[k] = 0
            depthCounts[k] += 1
            optMoveCounts += len(optimals)
            if gresult > 0:
                winMoves += 1
            elif gresult == 0:
                drawMoves += 1
            else:
                lossMoves += 1
        
        logMsg("Depth distribution is:")
        dk = sorted(dict.keys(depthCounts))
        for d in dk:
            logMsg(str(d) + ": " + str(depthCounts[d]))
        logMsg("Average number of correct moves is:", optMoveCounts / len(results))
        winMoves = 100.0 * (float(winMoves) / len(results))
        lossMoves = 100.0 * (float(lossMoves) / len(results))
        drawMoves = 100.0 * (float(drawMoves) / len(results))
        logMsg("Wins %.2f%%, Losses %.2f%%, Draws %.2f%%" % (winMoves, lossMoves, drawMoves))

        self.results = results

        self.package()

        for wproc in self.workerprocs:
            wproc.kill()

    def package(self):
        buffer = io.BytesIO()
        for path, optimals, gresult in self.results:
            buffer.write(bytes([49 + p for p in path]))
            buffer.write(bytes([32]))
            buffer.write(bytes([49 + o for o in optimals]))
            buffer.write(bytes([32]))
            buffer.write(bytes([49 + gresult]))
            buffer.write(bytes([10]))

        bys = buffer.getvalue()
        zipped = gzip.compress(bys)
        logMsg("Package size is %i kb, compressed to %i kb" % ((len(bys) // 1024), (len(zipped) // 1024)))
        with open(self.outputFile, "wb") as f:
            f.write(zipped)

