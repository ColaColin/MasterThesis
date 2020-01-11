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

    def test_fit(self):
        prntVerbose = ('-v' in sys.argv) or ('--verbose' in sys.argv)
        setLoggingEnabled(prntVerbose)

        # we just make up some target data for learning
        games = self.getRandomGameStates()

        reports = []
        policyUUID = str(uuid.uuid4())
        for gidx, game in enumerate(games):
            r = dict()
            r["knownResults"] = [gidx % (game.getPlayerCount() + 1)]
            r["policyIterated"] = np.zeros(game.getMoveCount(), dtype=np.float32)
            r["policyIterated"][gidx % game.getMoveCount()] = 1
            r["state"] = game.store()
            r["gamename"] = game.getGameName()
            r["uuid"] = str(uuid.uuid4())
            r["parent"] = str(uuid.uuid4())
            r["policyUUID"] = policyUUID
            reports.append(r)
        
        prevUUID = self.subject.getUUID()
        self.subject.fit(reports, 250)
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

