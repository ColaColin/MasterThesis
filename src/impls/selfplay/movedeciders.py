from impls.selfplay.LinearSelfPlay import SelfPlayMoveDecider

import abc

import numpy as np

from utils.prints import logMsg

class TemperatureMoveDecider(SelfPlayMoveDecider, metaclass=abc.ABCMeta):

    """
    if the game is in an early turn decide randomly on the moves proportional to the mcts policy
    if the game in not in an early turn deterministically play the stronges move found
    """

    def __init__(self, explorationPlyCount, minProp = -1):
        """
        @paramm explorationPlyCount: Until which ply (turn) to explore randomly
        """
        logMsg("Creating TemperatureMoveDecider(explorationPlyCount=%i)" % explorationPlyCount)
        self.explorationPlyCount = explorationPlyCount
        self.minProp = minProp

    def decideMove(self, gameState, policyDistribution, extraStats):
        legalMoves = np.array(gameState.getLegalMoves(), dtype=np.int)
        # shuffle so in case there are multiple moves with the same highest value 
        # the deterministic play actually picks one of them randomly,
        # instead of producing a bias to whatever move happens to be the lowest index one.
        np.random.shuffle(legalMoves)
        cleanPolicy = policyDistribution[legalMoves]

        if gameState.getTurn() > self.explorationPlyCount:
            chosenMove = legalMoves[np.argmax(cleanPolicy)]
        else:
            pSum = np.sum(cleanPolicy)
            assert pSum > 0
            cleanPolicy /= pSum
            if self.minProp > 0:
                for idx in range(len(cleanPolicy)):
                    if cleanPolicy[idx] < self.minProp:
                        cleanPolicy[idx] = 0
                pSum = np.sum(cleanPolicy)
                assert pSum > 0
                cleanPolicy /= pSum
            chosenMove = np.random.choice(legalMoves, 1, replace=False, p = cleanPolicy)[0]

        #     print(gameState.prettyString(policyDistribution, cleanPolicy, None, None))

        #     for m, p in zip(legalMoves, cleanPolicy):
        #         print(m, p)
            
        #     print("====>")
        #     print(chosenMove)

        # print("=====")

        assert chosenMove in legalMoves

        return chosenMove