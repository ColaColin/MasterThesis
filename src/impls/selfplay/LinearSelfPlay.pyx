# cython: profile=False

from core.selfplay.SelfPlayWorker import SelfPlayWorker

import uuid

import abc

class SelfPlayMoveDecider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decideMove(self, gameState, policyDistribution, extraStats):
        """
        @return move to be played based on the gameState and the policy and extra stats found by the policy iterator.
        """

class LinearSelfPlayWorker(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initalState, policy, policyIterator, gameCount, moveDecider):
        super().__init__(initalState, policy, policyIterator)
        self.initalState = initalState;
        self.open = [initalState] * gameCount
        self.tracking = [None] * gameCount;
        self.moveDecider = moveDecider

    def selfplay(self, gameReporter, policyUpdater):
        while True:
            self.finalizeGames()

            iteratedPolicy = self.policyIterator.iteratePolicy(self.policy, self.open)

            self.addTrackingData(gameReporter, iteratedPolicy)

            movesToPlay = map(lambda x: self.moveDecider.decideMove(x[0], x[1][0], x[1][1]), zip(self.open, iteratedPolicy))
           
            self.open = map(lambda state, move: state.playMove(move), zip(self.open, movesToPlay))

            self.policy = policyUpdater.update(self.policy)

    def addTrackingData(self, gameReporter, iteratedPolicy):
        for idx, game in enumerate(self.open):
            if self.tracking[idx] == None:
                self.tracking[idx] = []
            
            self.tracking[idx].append([game, iteratedPolicy[idx]])
            
        reports = []
        for idx in range(len(self.tracking)):
            trackList = self.tracking[idx]
            if trackList[len(trackList)-1][0].hasEnded():
                result = trackList[len(trackList)-1][0].getWinnerNumber()
                policyUUID = self.policy.getUUID()

                prevStateUUID = None
                for state, iPolicy in trackList:
                    # do not learn from terminal states, there is no move that can be made on them
                    if not state.hasEnded():
                        record = dict()
                        record["knownResults"] = [result]
                        record["generics"] = dict(iPolicy[1])
                        record["policyIterated"] = iPolicy[0]
                        record["uuid"] = str(uuid.uuid4())
                        record["parent"] = prevStateUUID
                        prevStateUUID = record["uuid"]
                        record["policyUUID"] = policyUUID
                        record["state"] = state.store()
                        record["gamename"] = state.getGameName()
                        reports.append(record)
                
                self.tracking[idx] = None

        if len(reports) > 0:
            gameReporter.reportGame(reports)

    def finalizeGames(self):
        """
        replace games that have been completed with new games, if desired by the impl.
        After this call all games in self.open should not be in a terminal state.
        The default impl just filters games out and replaces them with new games.
        """
        self.open = map(lambda x: self.initalState if x.hasEnded() else x, self.open)



