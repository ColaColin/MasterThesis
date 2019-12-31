import abc

class PolicyIterator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def iteratePolicy(self, policy, gamesBatch):
        """
        @param policy: a policy as defined by the Policy interface
        @param gamesBatch: is a list of game objects as defined by the GameState interface
        @return: 
        a list of tuples: the first element is a 1d float array representing the new policy over the moves to play
        the second element contains other data that is tracked in a dict, but not yet defined clearly (TODO)
        One element it has to contain is "policyRaw", which is the network output for the state. The entire tensor, not just the move probabilities!
        Return values should be save to keep around, so probably best to copy them before returning any internal data structure
        """