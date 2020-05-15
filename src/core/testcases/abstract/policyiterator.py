import abc
import sys
import numpy as np
import uuid

from core.base.GameState import GameState
from core.base.Policy import Policy

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

    def getGameConstructorName(self):
        return "core.testcases.abstract.policyiterator.TestCountGame"

    def getGameConstructorParams(self):
        return {"playersCount": self.playersCount, "pathLength": self.pathLength, "movesCount": self.movesCount, "resetOnError": self.resetOnError}

    def getGameName(self):
        return "TestGame"

    def getPlayerOnTurnNumber(self):
        return (self.turn % self.playersCount) + 1

    def mapPlayerNumberToTurnRelative(self, number):
        assert False, "no test calls this to my knowledege"

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
        return (3, )
    
    def encodeIntoTensor(self, tensor, batchIndex, augment):
        """
        the TestGame tells the player who will win as well as what move to play,
        depending on what the test policy implementation does what this 
        it is possible to implement oracle policies to check how the 
        policyiterator makes use of given information from the policy
        """
        curPlayer = self.getPlayerOnTurnNumber()
        curProgress = self.playersProgress[curPlayer]
        tensor[batchIndex][0] = self.winningPath[curProgress]

        bestPlayer = curPlayer
        bestScore = curProgress

        allEven = True
        for pid in range(1, self.playersCount+1):
            pProgress = self.playersProgress[pid]
            if pProgress != curProgress:
                allEven = False
            if pProgress > bestScore:
                bestScore = pProgress
                bestPlayer = pid
        
        tensor[batchIndex][1] = self.getPlayerOnTurnNumber() if allEven else bestPlayer

        tensor[batchIndex][2] = curProgress
    
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
    
    def prettyString(self, networkMoves, networkWins, iteratedMoves, observedWins):
        return str(self)

    def __str__(self):
        result = "TestGame with %i players, %i moves, %i pathLength, resetOnError: %i\n" % (self.playersCount, self.movesCount, self.pathLength, self.resetOnError)
        result += "Winning path is: " + str(self.winningPath) + "\n"
        result += "Player progress is: " + str(self.playersProgress) + "\n"
        if self.winner == -1:
            result += "In turn " + str(self.turn) + " with player " + str(self.getPlayerOnTurnNumber()) + "\n"
        else:
            result += "Winner is " + str(self.winner) + "\n"
        return result

class OraclePolicy(Policy, metaclass=abc.ABCMeta):
    """
    OraclePolicy can perfectly predict the winner and/or moves for TestCountGame.
    It will be a random policy, if told not to predict anything at all.
    """
    def __init__(self, moves, players, protoState, predictMoves, predictWinners, errorFrom=9999):
        self.uuid = uuid.uuid4()
        self.moves = moves
        self.players = players
        self.protoState = protoState
        self.predictMoves = predictMoves
        self.errorFrom = errorFrom
        self.predictWinners = predictWinners
    
    def forward(self, batch, asyncCall = None):
        result = []

        if self.predictMoves or self.predictWinners:
            tensors = np.zeros((len(batch), ) + self.protoState.getDataShape(), dtype=np.int32)

        for idx in range(len(batch)):
            if self.predictMoves or self.predictWinners:
                batch[idx].encodeIntoTensor(tensors, idx, False)

            if self.predictMoves and self.errorFrom > tensors[idx,2]:
                m = np.zeros(self.moves, dtype=np.float32)
                m[tensors[idx,0]] = 1
            else:
                m = np.random.uniform(size = self.moves).astype(np.float32)
                m /= np.sum(m)
            
            if self.predictWinners and self.errorFrom > tensors[idx,2]:
                w = np.zeros(self.players + 1, dtype=np.float32)
                w[tensors[idx,1]] = 1
            else:
                w = np.random.uniform(size = self.players + 1).astype(np.float32)
                w /= np.sum(w)

            result.append((m, w))

        if not (asyncCall is None):
            asyncCall()

        return result

    def getUUID(self):
        return self.uuid

    def prepareExample(self, frame):
        assert False, "Not implemented"

    def packageExamplesBatch(self, examples):
        assert False, "Not implemented"

    def getExamplePrepareObject(self):
        assert False, "Not implemented"

    def fit(self):
        assert False, "Not implemented"

    def reset(self):
        assert False, "Not implemented"

    def load(self, packed):
        assert False, "Not implemented"

    def store(self):
        assert False, "Not implemented"


class TestPolicyIterationSanity(metaclass=abc.ABCMeta):
    """
    Verifes the PolicyIterator impl is able to find obviously good moves when searching enough with a random policy 
    that does not help at all. The tests are probabilistic and can fail occasionally, which is why they test
    for rather low barriers, in most cases they can pass with much higher limits set.
    """

    @abc.abstractmethod
    def setUp(self):
        """
        Setup a field subject with a PolicyIterator implementation to be tested for sane behavior.
        """

    def getTestResult(self, players, moves, pathLength, resetOnError, batchSize, predictMoves=False, predictWins=False, errorFrom=9999):
        games = []
        for _ in range(batchSize):
            games.append(TestCountGame(players, pathLength, moves, resetOnError))
        policy = OraclePolicy(games[0].getMoveCount(), games[0].getPlayerCount(), games[0], predictMoves, predictWins, errorFrom=errorFrom)
        winningMoves = [game.winningPath[0] for game in games]
        iterated = self.subject.iteratePolicy(policy, games)
        self.assertEqual(len(iterated), batchSize)
        return iterated, winningMoves

    def verifyTestResult(self, testResult, requiredPercentage, errorsAllowed):
        iterated, winningMoves = testResult
        errors = 0
        for i in range(len(iterated)):
            isOK = iterated[i][0][winningMoves[i]] > requiredPercentage
            if not isOK:
                errors += 1
            if errors > errorsAllowed:
                self.assertTrue(False, "A winning move only got: " + str(iterated[i][0][winningMoves[i]]) + ", but required: " + str(requiredPercentage))
                break

    def test_2Players2Moves1Path5Batch(self):
        """
        the most simple possible case: pick the right move to win instantly
        """
        testResult = self.getTestResult(2, 2, 1, True, 5)
        self.verifyTestResult(testResult, 0.75, 2)

    def test_3Players3Moves2Path5Batch(self):
        """
        a more complex case, more players, more moves, need to hit two moves correctly in row
        """
        testResult = self.getTestResult(3, 3, 2, True, 5)
        self.verifyTestResult(testResult, 0.65, 2)
        
    def test_2Players8Moves1Path5Batch(self):
        """
        many moves
        """
        testResult = self.getTestResult(2, 8, 1, True, 5)
        self.verifyTestResult(testResult, 0.5, 2)

    def test_usesMoveOraclePolicyCorrectly(self):
        """
        Verify the policy iterator can make use of information provided in the moves predicition for states,
        in face of random winner predicition
        """
        testResult = self.getTestResult(2, 2, 12, True, 5, predictMoves=True, errorFrom=9)
        self.verifyTestResult(testResult, 0.75, 0)

    def test_usesWinnerOraclePolicyCorrectly(self):
        """
        Verify the policy iterator can make use of information provided in the winner predicition for states,
        in face of random move predicition
        """
        testResult = self.getTestResult(2, 2, 12, True, 5, predictWins=True, errorFrom=10)
        self.verifyTestResult(testResult, 0.75, 1)



