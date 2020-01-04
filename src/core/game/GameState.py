import abc

class GameState(metaclass=abc.ABCMeta):
    """
    Creating a new GameState object should initialize a new game to be played from the starting position.
    GameState objects are immutable, playMove produces a copy.
    """

    @abc.abstractmethod
    def getGameName(self):
        """
        @return: The name of the game
        """

    @abc.abstractmethod
    def getPlayerOnTurnNumber(self):
        """
        @return the number of the player on turn. Player numbers start at 1.
        """

    @abc.abstractmethod
    def hasEnded(self):
        """
        @return: True iff this state represents a game that has ended, no more moves can be played.
        """

    @abc.abstractmethod
    def getWinnerNumber(self):
        """
        @return: the number of the player who has won this game. Player numbers start at 1.
        0 implies the game is a Draw. Undefined behavior until hasEnded() returns true
        """

    @abc.abstractmethod
    def getLegalMoves(self):
        """
        @return: a list of indices of moves that can be played by the player on turn. Move indices are always between 0 and getMoveCount().
        A move with a given index should always refer the same action in the game.
        """

    @abc.abstractmethod
    def getPlayerCount(self):
        """
        @return: The number of players playing the game
        """

    @abc.abstractmethod
    def getMoveCount(self):
        """
        @return: how many possible moves there in this game. Important for the encoding of all existing moves in a probability distribution
        """

    @abc.abstractmethod
    def playMove(self, legalMoveIndex):
        """
        @return: A deep copy of this state with the given move played by the player who is currently on turn
        """

    @abc.abstractmethod
    def getTurn(self):
        """
        return the number of moves that have been played in this game state so far
        """

    @abc.abstractmethod
    def getDataShape():
        """
        returns the shape of the game data encoding for policy processing (e.g. the neural net input without the batch dimension, which is the first dimension)
        e.g. (2, width, height) or similar
        """

    @abc.abstractmethod
    def encodeIntoTensor(self, tensor, batchIndex, augment):
        """
        Write floats that represent the game situation into the given tensor
        @param tensor: A tensor object (numpy!) of shape (batchIndex, ) + getDataShape()
        @param batchIndex: Use this to write to the specific batch index
        @param augment: If true the data may be augmented randomly by whatever means the GameState implementation thinks fit the game (rotation, mirroring, etc)
        """

    @abc.abstractmethod
    def store(self):
        """
        @return: A representation of the GameState as a byte array (numpy) for external storage and later usage via the static load()
        """

    @abc.abstractmethod
    def load(self, encoded):
        """
        Create a new GameState object from an encoded representation produced by store()
        @param encoded: The encoded representation. Typically a byte array.
        """

    @abc.abstractmethod
    def __eq__(self, other):
        """
        Compare this GameState for equality to another
        """

    @abc.abstractmethod
    def __hash__(self):
        """
        Calculate a hash for this GameState
        """

    @abc.abstractmethod
    def __str__(self):
        """
        Produce a human readable string for debugging purposes
        """