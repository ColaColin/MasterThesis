import abc

class PIteratorInstance(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getResult(self):
        """
        @return
        a list of tuples: the first element is a 1d float array representing the new policy over the moves to play
        the second element contains other data that is tracked in a dict,
        which can be anything the policy iterator might want to show to the outside world (e.g. useful for debugging).
        Return values should be save to keep around, so probably best to copy 
        them before returning any internal data structure
        """
    
    @abc.abstractmethod
    def getCurrentPlayerParameters(self):
        """
        return the hyperparameters used by the current player
        """

    @abc.abstractmethod
    def getGame(self):
        """
        @return the game this iterator is for
        """

# in hindsight this class should have been named PolicyIteratorFactory, but the name is stuck now
class PolicyIterator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def createIterator(self, game, playerHyperparametersDict=dict(), noExploration=False):
        """
        @param noExploration: set to True to tell the iterator to minimize all random influences on the results, preferably to zero. Meant for evaluation purposes.
        @param game: A game instance
        create an iterator instance for the given game.
        @param playerHyperparametersDict: a dict of player numbers to hyperparameter configus to use to play that player
        {1->dict, 2->dict}. If no entry for a player number, use the standard parameters for that player.
        @return: A PIteratorInstance
        """

    def iteratePolicy(self, policy, gamesBatch, noExploration = False, playerHyperparametersDict=dict()):
        """
        old iteratePolicy, still used in places where fine grained control is not required (i.e. everywhere except PlayersSelfplay).
        Handles iterator creation for the user, but also falls back to calling the extended version internally.
        """
        iterators = [self.createIterator(g, playerHyperparametersDict=playerHyperparametersDict, noExploration=noExploration) for g in gamesBatch]
        self.iteratePolicyEx(policy, iterators)
        return [ix.getResult() for ix in iterators]

    @abc.abstractmethod
    def iteratePolicyEx(self, policy, iterators, iterations = None):
        """
        extended version of iteratePolicy, which needs the user to create iterators via createIterator by hand before calling this method.
        @param policy: a policy as defined by the Policy interface
        @param iterators: is a list of iterator objects, created by createIterator
        @param iterations: number of iterations to do. If None use whatever is standard for the PolicyIterator.
        @return: nothing, call getResult() on the given iterators instead
        """