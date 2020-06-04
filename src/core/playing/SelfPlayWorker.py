import abc

class GameReporter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reportGame(self, reports):
        """
        @param reports: A list of game state recordings. A game state recording is a dict with the following properties (there is no actual GameState object here!):

        gameCtor: name of the constructor of the played game
        gameParams: params to the constructor of the played game
        knownResults: list of player number of the winner that were reached from this state.
            For a simple self playing implementation there will be exactly one entry here.
        generics: a dict() of extra data that may be useful to some specialized analysis tools
        policyIterated: The move probabilities, improved by MCTS
        uuid: A UUID identifying the state
        parent: UUID of the parent state (None if this is a root)
        policyUUID: the UUID of the policy that was used to play this state
        state: store() of the actual game state. Numpy array.
        gamename: getGameName() of the game

        This call takes ownership of the given reports and might modify them for the purpose of storing them.
        """

class PolicyUpdater(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, policy):
        """
        Should update the given policy
        @return: Return a new policy to be used. Can be the old one if no changes were made.
        """

class SelfPlayWorker(metaclass=abc.ABCMeta):
    """
    This interface is only really important for the frametime measurement worker,
    """

    @abc.abstractmethod
    def main(self):
        """
        Play games against yourself forever.
        """

    @abc.abstractmethod
    def initSelfplay(self, runId):
        """
        Call once before starting to loop over playBatch()
        """

    @abc.abstractmethod
    def playBatch(self):
        """
        call to play one move on every game in the batch.
        Returns time taken for a single move in ms, excluding time spent doing game reporting or policy updates.
        """
