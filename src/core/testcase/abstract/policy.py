import abc
import numpy as np

from core.testcase.abstract.gamestate import playRandomMove

class TestPolicySanity(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def setUp(self):
        """
        Setup a field subject with a Policy implementation to be tested for sane behavior.
        The policy should be able to handle a batchSize of at least 10
        """

    @abc.abstractmethod
    def getExampleGameState(self):
        """
        Return an example game state
        """

    def getRandomGameStates(self):
        n = 10
        games = [self.getExampleGameState()] * n
        for i in range(n):
            for idx in range(5):
                games[i] = playRandomMove(games[i], idx, i)
        return games

    def forwardToLists(self, games):
        f = self.subject.forward(games)
        return [(r.tolist(), w.tolist()) for (r, w) in f]

    def test_loadStore(self):
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

    