import abc

class PolicyIterator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def iteratePolicy(self, policy, gamesBatch):
        """
        @param policy: a policy as defined by the Policy interface
        @param gamesBatch: is a list of game objects as defined by the GameState interface
        @return: a probability distribution over all moves for every game state in the batch (so an array of 1d float arrays). These represent examples of a would-be better policy than the one given, that should be learned from by the given policy for the core learning loop of Alpha Zero.
        """

    