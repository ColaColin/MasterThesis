import abc

class GameReporter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reportGame(self, reports):
        """
        @param reports: A list of game states. A game state is a dict with the following properties:
        knownResults: list of player number of the winner
        policyRaw: the tensor output of the policy (assuming a neural net)
        policyIterated: The move probabilities, improved by MCTS
        uuid: A UUID identifying the state
        parent: UUID of the parent state (None if this is a root)
        policyUUID: the UUID of the policy that was used to play this state
        """

class PolicyUpdater(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, policy):
        """
        Should update the given policy
        """

class SelfPlayWorker(metaclass=abc.ABCMeta):

    def __init__(self, initialState, policy, policyIterator):
        self.initialState = initialState
        self.policy = policy
        self.policyIterator = policyIterator

    @abc.abstractmethod
    def selfplay(self, gameReporter, policyUpdater):
        """
        Calling this should play games and call the gameReporter with new records of games to be learned from.
        Additionally call the PolicyUpdater in a regular fashion. That might pull a new policy from
        a training server or it might train the policy on data gathered so far.
        """
