import abc
import numpy as np
import uuid
import sys

from core.testcases.abstract.gamestate import playRandomMove

from utils.prints import logMsg, setLoggingEnabled

class TestPolicySanity(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def setUp(self):
        """
        Setup a field subject with a Policy implementation to be tested for sane behavior.
        """

    @abc.abstractmethod
    def getExampleGameState(self):
        """
        Return an example game state
        """

    def getRandomGameStates(self):
        n = 15
        games = [self.getExampleGameState()] * n
        for i in range(n):
            for idx in range(10):
                games[i] = playRandomMove(games[i], idx, i)

        dedupe = dict()
        for g in games:
            dedupe[g] = g

        return list(dedupe.values())

    def forwardToLists(self, games):
        f = self.subject.forward(games)
        return [(r.tolist(), w.tolist()) for (r, w) in f]

    def packageGameAsFrame(self, game, gidx, policyUUID):
        r = dict()
        r["knownResults"] = [gidx % (game.getPlayerCount() + 1)]
        r["policyIterated"] = np.zeros(game.getMoveCount(), dtype=np.float32)
        r["policyIterated"][gidx % game.getMoveCount()] = 1
        r["state"] = game.store()
        r["gamename"] = game.getGameName()
        r["uuid"] = str(uuid.uuid4())
        r["parent"] = str(uuid.uuid4())
        r["policyUUID"] = policyUUID
        return r

    def makeEqualExamples(self):
        gameA = self.getExampleGameState()
        gameB = self.getExampleGameState()
        for i in range(3):
            gameA = playRandomMove(gameA, 0, i)
            gameB = playRandomMove(gameB, 0, i)
        
        frameA = self.packageGameAsFrame(gameA, 0, self.subject.getUUID())
        frameB = self.packageGameAsFrame(gameB, 1, self.subject.getUUID())

        batcher = self.subject.getExamplePrepareObject()
        exampleA = batcher.prepareExample(frameA)
        exampleB = batcher.prepareExample(frameB)

        return exampleA, exampleB

    def test_HashFits(self):
        prntVerbose = ('-v' in sys.argv) or ('--verbose' in sys.argv)
        setLoggingEnabled(prntVerbose)

        exampleA, exampleB = self.makeEqualExamples()
        batcher = self.subject.getExamplePrepareObject()
        self.assertEqual(batcher.getHashForExample(exampleA), batcher.getHashForExample(exampleB))

    def test_areExamplesEqual(self):
        prntVerbose = ('-v' in sys.argv) or ('--verbose' in sys.argv)
        setLoggingEnabled(prntVerbose)

        exampleA, exampleB = self.makeEqualExamples()
        batcher = self.subject.getExamplePrepareObject()
        self.assertTrue(batcher.areExamplesEqual(exampleA, exampleB))

    def test_fit(self):
        prntVerbose = ('-v' in sys.argv) or ('--verbose' in sys.argv)
        setLoggingEnabled(prntVerbose)

        # we just make up some target data for learning
        games = self.getRandomGameStates()

        reports = []
        policyUUID = str(uuid.uuid4())
        for gidx, game in enumerate(games):
            reports.append(self.packageGameAsFrame(game, gidx, policyUUID))
        
        prevUUID = self.subject.getUUID()
        preparedReports = [self.subject.prepareExample(report) for report in reports]
        for e in range(250):
            self.subject.fit(self.subject.packageExamplesBatch(preparedReports), e, 1)

        self.assertNotEqual(prevUUID, self.subject.getUUID())

        forwards = self.forwardToLists(games)

        for ridx, report in enumerate(reports):
            winnerIndex = report["knownResults"][0]
            moveIndex = np.argmax(report["policyIterated"])
            forward = forwards[ridx]
            self.assertEqual(np.argmax(forward[0]), moveIndex)
            self.assertEqual(np.argmax(forward[1]), winnerIndex)

    def test_loadStore(self):
        prntVerbose = ('-v' in sys.argv) or ('--verbose' in sys.argv)
        setLoggingEnabled(prntVerbose)

        games = self.getRandomGameStates()
        x = self.subject.store()

        forwards = self.forwardToLists(games)

        originalUUID = self.subject.getUUID()
        self.subject.reset()

        self.assertNotEqual(forwards, self.forwardToLists(games))
        self.assertNotEqual(originalUUID, self.subject.getUUID())

        self.subject.load(x)
        self.assertEqual(self.subject.getUUID(), originalUUID)
        self.assertEqual(forwards, self.forwardToLists(games))

