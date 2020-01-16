import abc

class GameSolver(metaclass=abc.ABCMeta):
    
    """
    represents a strong solver for a game, meant to generate test datasets for
    solved games. 
    """

    @abc.abstractmethod
    def getMoveScores(self, state, movesReplay):
        """
        @return: A dict that uses legal moves as keys and assigns a score to each move.
        The score should be positive for winning moves, negative for
        losing moves and the highest score represents optimal play.
        """