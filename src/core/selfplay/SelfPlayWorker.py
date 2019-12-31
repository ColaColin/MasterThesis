import abc

class SelfPlayWorker(metaclass=abc.ABCMeta):
    def __init__(self, initalState, policy, policyIterator, startCount):
        self.startCount = startCount
        self.initalState = initalState;
        self.open = [initalState] * startCount
        self.policy = policy;
        self.policyIterator = policyIterator

    def selfplay(self):
        while self.shouldContinueSelfplay():
            self.finalizeGames()

            iteratedPolicy = self.policyIterator.iteratePolicy(self.policy, self.open)
            movesToPlay = map(lambda x: self.decideMove(x[0], x[1][0], x[1][1]), zip(self.open, iteratedPolicy))
            self.prevStates = self.open
            self.open = map(lambda x: x[0].playMove(x[1]), zip(self.open, movesToPlay))
            # CONTINUE so what should be tracked here exactly?

    def finalizeGames(self):
        """
        replace games that have been completed with new games, if desired by the impl.
        After this call all games in self.open should not be in a terminal state.
        The default impl just filters games out and replaces them with new games.
        """
        self.open = map(lambda x: self.initalState if x.hasEnded() else x, self.open)


    @abc.abstractmethod
    def decideMove(self, gameState, policyDistribution, winProbs):
        """
        @return the move to be played based on the gameState and the policy and win probs found by the policy iterator
        """

    def shouldContinueSelfplay(self):
        return True