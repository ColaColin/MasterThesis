import abc
import sys
import numpy as np

def playRandomMove(game, idx, offset):
    if not game.hasEnded():
        moves = game.getLegalMoves()
        move = moves[(idx + offset) % len(moves)]
        return game.playMove(move)
    else:
        return game

class TestGameStateSanity(metaclass=abc.ABCMeta):
    """
    Base class to test game implementations for simple sanity tests and
    a the outcome of a few example games.
    """

    @abc.abstractmethod
    def setUp(self):
        """
        Setup a field subject with a GameState implementation to be tested for sane behavior
        """

    @abc.abstractmethod
    def getExampleGameSequences(self):
        """
        return a list of tuples that represent games:
        ([list of moves played], finalTurnNumber, winningPlayer)
        A test will check if all these games work out to the expected finalTurnNumber and winningPlayer.
        The example games have to end in a terminal state!
        """

    def printGameWithTensor(self, game):
        tensor = np.zeros((1,) + game.getDataShape(), dtype=np.float32)
        game.encodeIntoTensor(tensor, 0, False)
        gs = str(game)
        ts = str(tensor)
        print(gs, ts)        

    def test_ExampleGames(self):
        """
        provided examples should play out as expected
        """
        prntVerbose = ('-v' in sys.argv) or ('--verbose' in sys.argv)

        def playGameByMoves(moves):
            game = self.subject
            if prntVerbose:
                self.printGameWithTensor(game)
            for move in moves:
                if game.hasEnded():
                    break
                game = game.playMove(move)
                if prntVerbose:
                    self.printGameWithTensor(game)
            return game

        examples = self.getExampleGameSequences()

        if prntVerbose:
            print("\nPlaying", len(examples), "examples")

        for idx, (moves, expectedTurns, expectedWinner) in enumerate(examples):
            if prntVerbose:
                print("Playing example game", idx)
            resultState = playGameByMoves(moves)
            self.assertEqual(resultState.getTurn(), expectedTurns)
            self.assertTrue(resultState.hasEnded())
            self.assertEqual(resultState.getWinnerNumber(), expectedWinner)

    def test_initialSubjectIsNotEnded(self):
        """
        The subject game should not be over before any players have made moves
        """
        self.assertFalse(self.subject.hasEnded())

    def test_gameEndsEventually(self):
        """
        At most 10000 turns should be plenty to reach a termainl state in any game that is used with AlphaZero.
        No matter how the players play.
        """
        maxMoves = 10000
        for i in range(maxMoves):
            if self.subject.hasEnded():
                break
            legalMoves = self.subject.getLegalMoves()
            self.assertTrue(len(legalMoves) > 0, "Suject offers no legal moves, but is not ended!")
            move = legalMoves[i % len(legalMoves)]
            self.subject = self.subject.playMove(move)
        self.assertTrue(self.subject.hasEnded())

    def test_turnsCountUp(self):
        """
        The turn counter should count the turns played
        """
        for i in range(100):
            self.assertEqual(self.subject.getTurn(), i)
            if self.subject.hasEnded():
                break
            self.subject = playRandomMove(self.subject, i, 0)

    def generateEq3Games(self):
        gameA = self.subject
        gameB = self.subject
        gameC = self.subject

        for i in range(42):
            gameA = playRandomMove(gameA, i, 0)
            gameB = playRandomMove(gameB, i, 0)
            gameC = playRandomMove(gameC, i, 1)
        
        return gameA, gameB, gameC

    def test_eqMatch(self):
        """
        Two games played with identical moves should be equal
        A third game with different moves should not be equal to the other two
        """

        gameA, gameB, gameC = self.generateEq3Games()

        self.assertEqual(gameA, gameB, "gameA == gameB")
        self.assertNotEqual(gameA, gameC, "gameA != gameC")
        self.assertNotEqual(gameB, gameC, "gameB != gameC")

    def test_hashMatch(self):
        """
        Two games played with identical moves should have the same hash
        A third game with different moves should not be equal to the other two
        """

        gameA, gameB, gameC = self.generateEq3Games()

        self.assertEqual(hash(gameA), hash(gameB), "hash(gameA) == hash(gameB)")
        self.assertNotEqual(hash(gameA), hash(gameC), "hash(gameA) != hash(gameC)")
        self.assertNotEqual(hash(gameB), hash(gameC), "hash(gameB) != hash(gameC)")

    def playRandomGame(self, idx):
        game = self.subject
        results = []
        for i in range(idx % 50):
            game = playRandomMove(game, i, idx * 7919)
            results.append(game)
            if game.hasEnded():
                break
        return results

    def test_hashProperties(self):
        """
        When playing random games there should be less than 20% hash collisions and 
        at most 16 states that share a single hash in the generated states.
        """
        numTestGames = 250
        states = map(lambda x: self.playRandomGame(x), range(numTestGames))
        statesByHash = dict()
        uniqueStates = 0
        oCnt = 0
        allStates = []
        worstLen = 0
        worstHashValue = 0
        for ss in states:
            for s in ss:
                allStates.append(s)
                oCnt += 1
                h = hash(s)
                if not h in statesByHash:
                    statesByHash[h] = [s]
                    uniqueStates += 1
                else:
                    isKnownState = len(list(filter(lambda x: x == s, statesByHash[h]))) > 0
                    if not isKnownState:
                        statesByHash[h].append(s)
                        if len(statesByHash[h]) > worstLen:
                            worstLen = len(statesByHash[h])
                            worstHashValue = h
                        uniqueStates += 1

        for aIdx in range(len(allStates)):
            for bIdx in range(aIdx+1, len(allStates)):
                a = allStates[aIdx]
                b = allStates[bIdx]
                if a == b:
                    self.assertEqual(hash(a), hash(b), "Equality must imply equal hash values")

        uniqueHashes = len(statesByHash)
        dupes = uniqueStates - uniqueHashes
        result = dupes / float(uniqueStates)

        if ('-v' in sys.argv) or ('--verbose' in sys.argv):
            print("\nFound ", uniqueHashes, "unique hashs for", uniqueStates, "unique states. Overall ", oCnt, "moves played! Worst hash has", worstLen, "collisions, it is the hash number", worstHashValue)
        self.assertTrue(uniqueHashes <= uniqueStates)
        self.assertTrue(result < 0.2)
        self.assertTrue(worstLen < 17)

    def test_storeLoad(self):
        """
        Store/Load should recreate equal games.
        """
        gameA, gameB, gameC = self.generateEq3Games()

        aStore = gameA.store()
        cStore = gameC.store()

        aLoaded = self.subject.load(aStore)
        cLoaded = self.subject.load(cStore)

        # first some tests on "known" sets of games, where we have two games known to be equal, without object identity
        self.assertNotEqual(aLoaded, self.subject, "aLoaded is not an empty game")
        self.assertNotEqual(cLoaded, self.subject, "cLoaded is not an empty game")

        self.assertEqual(gameA, aLoaded, "gameA == aLoaded")
        self.assertEqual(gameC, cLoaded, "gameC == cLoaded")
        self.assertEqual(hash(gameA), hash(aLoaded), "hash(gameA) == hash(aLoaded)")
        self.assertEqual(hash(gameC), hash(cLoaded), "hash(gameC) == hash(cLoaded)")

        numTestGames = 250
        states = map(lambda x: self.playRandomGame(x), range(numTestGames))
        for ss in states:
            for s in ss:
                stored = s.store()
                loaded = self.subject.load(stored)
                self.assertEqual(loaded, s)

    def test_storeLoadConsistent(self):
        """
        Calling store, load, store should produce the same bytes as the first call to store
        """
        numTestGames = 250
        states = map(lambda x: self.playRandomGame(x), range(numTestGames))
        for ss in states:
            for s in ss:
                stored = s.store()
                loaded = self.subject.load(stored)
                stored2 = loaded.store()
                self.assertEqual(stored.tolist(), stored2.tolist())