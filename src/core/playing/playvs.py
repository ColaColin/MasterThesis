import abc

class ExternalPlayerInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getMove(self, gameState):
        """
        Somehow get the move the external entity (e.g. a human or other AI) wants to play and return it.
        This method is expected to block while the other entity thinks.
        @param gameState: The game state to make a move on. For human players this can be ignored, as showGame is called right before.
        @return: a move index to be played. Must refer to a legal move.
            Illegal moves will cause the function to be called again.
        """

    @abc.abstractmethod
    def showGame(self, gameState):
        """
        Show the current state of the game to the (human) player. AIs likely don't need to do anything on this call.
        """

    @abc.abstractmethod
    def showFinalResult(self, externalId, gameState):
        """
        Called at the end of the game with the resulting game state
        """

class PlayVs():
    """
    Let some external entity (e.g. human or AI) play vs the AI in a N player game where 
    all players but one (none is also possible by setting some non-existent ID) are AI controlled.
    """

    def __init__(self, externalPlayerNumber, policy, policyIterator, policyUpdater, initialState, moveDecider, external):
        self.externalNumber = externalPlayerNumber
        self.currentState = initialState
        self.policy = policyUpdater.update(policy)
        self.policyIterator = policyIterator
        self.moveDecider = moveDecider
        self.external = external

    def playVs(self):
        while not self.currentState.hasEnded():
            if self.currentState.getPlayerOnTurnNumber() == self.externalNumber:
                self.external.showGame(self.currentState)
                move = self.external.getMove(self.currentState)
            else:
                self.external.showGame(self.currentState)
                iteratedPolicy = self.policyIterator.iteratePolicy(self.policy, [self.currentState])[0]
                move = self.moveDecider.decideMove(self.currentState, iteratedPolicy[0], iteratedPolicy[1])

            if move in self.currentState.getLegalMoves():
                self.currentState = self.currentState.playMove(move)

        self.external.showGame(self.currentState)
        self.external.showFinalResult(self.externalNumber, self.currentState)