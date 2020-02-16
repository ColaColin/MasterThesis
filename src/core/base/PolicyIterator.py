import abc

class PolicyIterator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def iteratePolicy(self, policy, gamesBatch, noExploration = False, quickFactor = 1):
        """
        @param policy: a policy as defined by the Policy interface
        @param gamesBatch: is a list of game objects as defined by the GameState interface
        @param noExploration: set to True to tell the iterator to minimize all random influences on the results, preferably to zero. Meant for evaluation purposes.
        @param quickFactor: divide the compute time put into the iteration by this number. Meant to reduce compute effort when running evaluation on a cpu system.
        @return: 
        a list of tuples: the first element is a 1d float array representing the new policy over the moves to play
        the second element contains other data that is tracked in a dict,
        which can be anything the policy iterator might want to show to the outside world (e.g. useful for debugging).
        Return values should be save to keep around, so probably best to copy 
        them before returning any internal data structure
        """