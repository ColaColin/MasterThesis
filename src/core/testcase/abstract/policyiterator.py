import abc
import sys
import numpy as np
import uuid

from core.game.GameState import GameState
from core.policy.Policy import Policy

class TestCountGame(GameState, metaclass=abc.ABCMeta):
    """
    N-Player game that requires the players to walk a random path that was decided when the game started.
    The game encodes how many moves you are along in the correct path (which is not used by any of the test use cases).
    It can be configured how many steps the path is long and how many possible directions there are to determine how hard it is to find the path.
    """

    def __init__(self, playersCount, pathLength, movesCount, resetOnError):
        """
        @param playersCount: How many players play
        @param pathLength: How many steps is the winning path long
        @param 
        """
        self.turn = 0
        self.playersCount = playersCount
        self.pathLength = pathLength
        self.movesCount = movesCount
        self.resetOnError = resetOnError

        self.playersProgress = dict()
        for i in range(playersCount):
            self.playersProgress[i+1] = 0
    
        self.legalMoves = [x + 1 for x in range(self.movesCount)]

        self.winningPath = []
        for i in range(self.pathLength):
            self.winningPath.append(np.random.randint(self.movesCount)+1)

        self.winner = -1

    def getGameName(self):
        return "TestGame"

    def getPlayerOnTurnNumber(self):
        return (self.turn % self.playersCount) + 1

    def hasEnded(self):
        return self.winner != -1

    def getWinnerNumber(self):
        return self.winner

    def getLegalMoves(self):
        return self.legalMoves.copy()

    def getPlayerCount(self):
        return self.playersCount

    def getMoveCount(self):
        """
        more moves than legal moves, just to test mcts behaved well with that
        """
        return self.movesCount + 2
    
    def playMove(self, move):
        if self.winner == -1:
            result = TestCountGame(self.playersCount, self.pathLength, self.movesCount, self.resetOnError)
            result.winningPath = self.winningPath.copy()            
            result.playersProgress = self.playersProgress.copy()
            result.turn = self.turn        

            curPlayer = result.getPlayerOnTurnNumber()
            curProgress = result.playersProgress[curPlayer]
            if curProgress < result.pathLength:
                if move == result.winningPath[curProgress]:
                    result.playersProgress[curPlayer] = curProgress + 1
                    if result.playersProgress[curPlayer] >= result.pathLength:
                        result.winner = curPlayer
                elif result.resetOnError:
                    result.playersProgress[curPlayer] = 0

            result.turn += 1

            return result
        else:
            return self
            
    def getTurn(self):
        return self.turn

    def getDataShape(self):
        return (1, )
    
    def encodeIntoTensor(self, tensor, batchIndex, augment):
        curPlayer = self.getPlayerOnTurnNumber()
        curProgress = self.playersProgress[curPlayer]
        tensor[batchIndex][0] = curProgress
    
    def store(self):
        fsize = self.playersCount + self.pathLength
        arSize = 6 + fsize

        result = np.zeros(arSize, dtype=np.int32)
        result[0] = self.turn
        result[1] = self.playersCount
        result[2] = self.pathLength
        result[3] = self.movesCount
        result[4] = 1 if self.resetOnError else 0
        result[5] = self.winner

        for i in range(self.playersCount):
            result[6 + i] = self.playersProgress[i+1]
        
        for i in range(self.pathLength):
            result[6 + self.playersCount + i] = self.winningPath[i]

        return result
    
    def load(self, encoded):
        turn = int(encoded[0])
        pCount = int(encoded[1])
        pLength = int(encoded[2])
        mCount = int(encoded[3])
        rError = encoded[4] == 1
        winner = int(encoded[5])

        result = TestCountGame(pCount, pLength, mCount, rError)
        result.turn = turn
        result.winner = winner
        for i in range(pCount):
            result.playersProgress[i+1] = int(encoded[6 + i])

        result.winningPath = []
        for i in range(pLength):
            result.winningPath.append(int(encoded[6 + pCount + i]))

        return result

    def __eq__(self, other):
        return self.winningPath == other.winningPath and \
            self.movesCount == other.movesCount and self.resetOnError == other.resetOnError and \
            self.playersProgress == other.playersProgress and self.winner == other.winner

    def __hash__(self):
        result = 0
        mul = 1
        for i in range(self.playersCount):
            result += self.playersProgress[i+1] * mul
            mul *= 10
        return result
    
    def __str__(self):
        result = "TestGame with %i players, %i moves, %i pathLength, resetOnError: %i\n" % (self.playersCount, self.movesCount, self.pathLength, self.resetOnError)
        result += "Winning path is: " + str(self.winningPath) + "\n"
        result += "Player progress is: " + str(self.playersProgress) + "\n"
        if self.winner == -1:
            result += "In turn " + str(self.turn) + " with player " + str(self.getPlayerOnTurnNumber()) + "\n"
        else:
            result += "Winner is " + str(self.winner) + "\n"
        return result

class RandomPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, moves, players):
        self.uuid = uuid.uuid4()
        self.moves = moves
        self.players = players

    def forward(self, batch, asyncCall = None):
        result = []
        for _ in batch:
            m = np.random.uniform(size = self.moves).astype(np.float32)
            w = np.random.uniform(size = self.players + 1).astype(np.float32)
            m /= np.sum(m)
            w /= np.sum(w)
            result.append((m, w))
        
        if not (asyncCall is None):
            asyncCall()

        return result

    def getUUID(self):
        return self.uuid

    def fit(self):
        pass

    def load(self, packed):
        assert False, "Not implemented"

    def store(self):
        assert False, "Not implemented"

class TestPolicyIterationSanity(metaclass=abc.ABCMeta):
    """
    Verifes the PolicyIterator impl is able to find obviously good moves when searching enough with a random policy 
    that does not help at all.
    """

    @abc.abstractmethod
    def setUp(self):
        """
        Setup a field subject with a PolicyIterator implementation to be tested for sane behavior.
        """

    def getTestResult(self, players, moves, pathLength, resetOnError, batchSize):
        game = TestCountGame(players, pathLength, moves, resetOnError)
        policy = RandomPolicy(game.getMoveCount(), game.getPlayerCount())
        winningMove = game.winningPath[0]
        iterated = self.subject.iteratePolicy(policy, [game] * batchSize)
        self.assertEqual(len(iterated), batchSize)
        return iterated, winningMove

    def verifyTestResult(self, testResult, requiredPercentage, errorsAllowed):
        iterated, winningMove = testResult
        errors = 0
        for i in range(len(iterated)):
            isOK = iterated[i][0][winningMove] > requiredPercentage
            if not isOK:
                errors += 1
            if errors > errorsAllowed:
                self.assertTrue(False, "A winning move only got: " + str(iterated[i][0][winningMove]) + ", but required: " + str(requiredPercentage))
                break

    def test_2Players2Moves1Path5Batch(self):
        """
        the most simple possible case: pick the right move to win instantly
        """
        testResult = self.getTestResult(2, 2, 1, True, 5)
        self.verifyTestResult(testResult, 0.7, 2)

    def test_2Players2Moves1Path10Batch(self):
        """
        the most simple cases, with a batch size of 10
        """
        testResult = self.getTestResult(2, 2, 1, True, 10)
        self.verifyTestResult(testResult, 0.7, 3)

    def test_3Players3Moves2Path5Batch(self):
        """
        a more complex case, more players, more moves, need to hit two moves correctly in row
        """
        testResult = self.getTestResult(3, 3, 2, True, 5)
        self.verifyTestResult(testResult, 0.7, 2)
        
    def test_2Players8Moves1Path5Batch(self):
        """
        many moves
        """
        testResult = self.getTestResult(2, 8, 1, True, 5)
        self.verifyTestResult(testResult, 0.7, 2)

    def test_2Players2Moves3Path5Batch(self):
        testResult = self.getTestResult(2, 2, 3, True, 5)
        self.verifyTestResult(testResult, 0.7, 2)
