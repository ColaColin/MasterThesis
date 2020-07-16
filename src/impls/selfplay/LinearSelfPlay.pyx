# cython: profile=False

from core.playing.SelfPlayWorker import SelfPlayWorker

import uuid

import abc

from utils.prints import logMsg

import time

import numpy as np
import datetime

import random

import torch
from torch.autograd import Variable
import sys
from core.training.NetworkApi import NetworkApi
from impls.polices.pytorch.policy import unpackTorchNetwork

class SelfPlayMoveDecider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decideMove(self, gameState, policyDistribution, extraStats):
        """
        @return move to be played based on the gameState and the policy and extra stats found by the policy iterator.
        """


def fillRecordForFeatures(featureProvider, reports, device):
    relevantPositions = set()

    for r in reports:
        relevantPositions.add(r["winFeatures"])
        relevantPositions.add(r["winFeatures+1"])
        relevantPositions.add(r["winFeatures+2"])
        relevantPositions.add(r["winFeatures+3"])

        relevantPositions.add(r["moveFeatures"])
        relevantPositions.add(r["moveFeatures+1"])
        relevantPositions.add(r["moveFeatures+2"])
        relevantPositions.add(r["moveFeatures+3"])

    rpositions = list(relevantPositions)

    if len(rpositions) > 0:
        winFeatures = dict()
        moveFeatures = dict()

        forwardShape = (len(rpositions), ) + rpositions[0].getDataShape()
        featuresNetInput = torch.zeros(forwardShape)
        npNetInput = featuresNetInput.numpy()
        for gidx, g in enumerate(rpositions):
            g.encodeIntoTensor(npNetInput, gidx, False)

        forwardInputGPU = Variable(featuresNetInput, requires_grad=False).to(device)
        with torch.no_grad():
            mOut, wOut, rOut, winFeaturesOut, moveFeaturesOut = featureProvider(forwardInputGPU)  

        for gidx, g in enumerate(rpositions):
            winFeatures[g] = winFeaturesOut[gidx]
            moveFeatures[g] = moveFeaturesOut[gidx]

        for r in reports:
            r["winFeatures"] = winFeatures[r["winFeatures"]].cpu().numpy().flatten().tolist()
            r["winFeatures+1"] = winFeatures[r["winFeatures+1"]].cpu().numpy().flatten().tolist()
            r["winFeatures+2"] = winFeatures[r["winFeatures+2"]].cpu().numpy().flatten().tolist()
            r["winFeatures+3"] = winFeatures[r["winFeatures+3"]].cpu().numpy().flatten().tolist()

            r["moveFeatures"] = moveFeatures[r["moveFeatures"]].cpu().numpy().flatten().tolist()
            r["moveFeatures+1"] = moveFeatures[r["moveFeatures+1"]].cpu().numpy().flatten().tolist()
            r["moveFeatures+2"] = moveFeatures[r["moveFeatures+2"]].cpu().numpy().flatten().tolist()
            r["moveFeatures+3"] = moveFeatures[r["moveFeatures+3"]].cpu().numpy().flatten().tolist()


class LinearSelfPlayWorker(SelfPlayWorker, metaclass=abc.ABCMeta):
    def __init__(self, initialState, policy, policyIterator, gameCount, moveDecider,\
            gameReporter, policyUpdater, capPolicyP = None, capPolicyIterator = None, capMoveDecider = None,\
            featureProvider = None, featureNetworkID = None):
        """
        To use playout cap randomization (see arXiv:1902.10565v3), configure:
        capPolicyP: what fraction of the games should be capped (this is 1-p in the paper, I find that more easy to understand)
        capPolicyIterator: policy iterator to use for capped play steps. E.g. MCTS with a lower expansions setting. Possibly also less cpuct, less fpu, no rootNoise, lower drawValue to reduce exploration and maximize strength in these moves.
        capMoveDecider: moveDecider to use on capped moves. Can be omitted, then the normal moveDecider will be used
        """
        logMsg("Creating LinearSelfPlayWorker gameCount=%i" % gameCount)
        self.initialState = initialState
        self.policy = policy
        self.policyIterator = policyIterator
        self.open = [initialState] * gameCount
        assert not initialState.hasEnded()
        self.tracking = [None] * gameCount;
        self.moveDecider = moveDecider
        self.microsPerMovePlayedHistory = []
        self.gameReporter = gameReporter
        self.policyUpdater = policyUpdater
        self.lastSpeedStatsPrint = time.monotonic()

        self.capPolicyP = capPolicyP
        self.capPolicyIterator = capPolicyIterator
        self.capMoveDecider = capMoveDecider

        assert (self.capPolicyP is None and self.capPolicyIterator is None) or (self.capPolicyP is not None and self.capPolicyIterator is not None), "capPolicyP and capPolicyIterator both need to be set or not set"

        if self.capPolicyP is not None:
            logMsg("Playout Cap Randomization is enabled, %i%% of moves will be played with a faster policy iterator and not recorded as training material to playout more games at little cost!" % (self.capPolicyP * 100))

        self.featureProvider = featureProvider
        # local feature networks are not supported, this always ends up loading via network.
        self.featureNetworkID = featureNetworkID

        if self.featureProvider is not None:
            logMsg("Using a feature providing network to add additional features to training data!")
            if self.featureNetworkID is not None:
                logMsg("Loading feature network %s" % self.featureNetworkID)
                networks = NetworkApi(noRun=True)
                networkData = networks.downloadNetwork(self.featureNetworkID)
                uuid, modelDict = unpackTorchNetwork(networkData)
                self.featureProvider.load_state_dict(modelDict)

            if torch.cuda.is_available():
                gpuCount = torch.cuda.device_count()
                device = "cuda"

                if "--windex" in sys.argv and gpuCount > 1:
                    windex = int(sys.argv[sys.argv.index("--windex") + 1])
                    gpuIndex = windex % gpuCount
                    device = "cuda:" + str(gpuIndex)
                    logMsg("Found multiple gpus with set windex, extended cuda device to %s" % device)

                self.device = torch.device(device)

                logMsg("Feature network will use the gpu!", self.device)
            else:
                logMsg("No GPU is available, falling back to cpu!")
                self.device = torch.device("cpu")
            
            self.featureProvider = self.featureProvider.to(self.device)
            self.featureProvider.train(False)

    def handleSpeedStats(self):
        if time.monotonic() - self.lastSpeedStatsPrint > 300:
            self.lastSpeedStatsPrint = time.monotonic()
            count = len(self.microsPerMovePlayedHistory)
            mean = np.mean(self.microsPerMovePlayedHistory)
            median = np.median(self.microsPerMovePlayedHistory)
            std = int(np.std(self.microsPerMovePlayedHistory))

            logMsg("Last %i loops move times: %i us median, %i us mean,  +/- %i us" % (count, median, mean, std))

            self.microsPerMovePlayedHistory = []

    def main(self):
        self.initSelfplay(None)

        while True:
            self.playBatch()

    def initSelfplay(self, runId):
        self.policy = self.policyUpdater.update(self.policy)
        assert not (self.policy is None), "the policy updater returned a None policy!"

    def playBatch(self):
        """
        For the frametime measurement, this has to measure how long it takes to generate a single frame of training data and return that.
        So if multiple moves are played per frame of training data, that has all to be measured.
        """
        moveTimeNanos = 0

        while True:
            playRealMove = self.capPolicyP is None or random.random() >= self.capPolicyP

            policyIterator = self.policyIterator
            moveDecider = self.moveDecider
            if not playRealMove:
                policyIterator = self.capPolicyIterator
                if self.capMoveDecider is not None:
                    moveDecider = self.capMoveDecider

            # generate iterated policy to be played on
            iteratationStart = time.monotonic_ns()
            iteratedPolicy = policyIterator.iteratePolicy(self.policy, self.open)
            moveTimeNanos += time.monotonic_ns() - iteratationStart

            if playRealMove:
                # remember the iterated policy for each game to be later processed into training frames.
                self.addTrackingData(iteratedPolicy)

                # potentially update the policy with a new policy from somewhere (e.g. a training server)
                self.initSelfplay(None)

            # play the moves using the iterated policy.
            playStart = time.monotonic_ns()
            movesToPlay = list(map(lambda x: moveDecider.decideMove(x[0], x[1][0], x[1][1]), zip(self.open, iteratedPolicy)))
            self.open = list(map(lambda x: x[0].playMove(x[1]), zip(self.open, movesToPlay)))
            moveTimeNanos += time.monotonic_ns() - playStart

            # go over finished games and replace them with new ones, reporting their frames to the server
            self.finalizeGames()

            if playRealMove:
                break

        # sum up the time used to iterate the policy and play the move
        usTime = int((moveTimeNanos / float(len(self.open))) / 1000)
        self.microsPerMovePlayedHistory.append(usTime)
        self.handleSpeedStats()

        return (usTime / 1000.0), None, len(self.open)

    def addTrackingData(self, iteratedPolicy):
        for idx, game in enumerate(self.open):
            if self.tracking[idx] == None:
                self.tracking[idx] = []
            
            self.tracking[idx].append([game, iteratedPolicy[idx]])

    def handleReportsFor(self, idx):
        reports = []

        trackList = self.tracking[idx]

        # this can happen with playout cap randomization, if an entire game is played with the cheap policy.
        # that tends to happen espacially with an untrained network that plays very bad moves.
        if trackList is None:
            return

        assert self.open[idx].hasEnded()
        result = self.open[idx].getWinnerNumber()
        policyUUID = self.policy.getUUID()

        prevStateUUID = None

        def getNextState(tlist, ti, offset):
            # this only supports two player games!
            while ti + offset * 2 >= len(tlist):
                offset -= 1
            
            assert offset >= 0
            
            state, iPolicy = tlist[ti + offset * 2]
            return state


        for ti in range(len(trackList)):
            state, iPolicy = trackList[ti]
            # do not learn from terminal states, there is no move that can be made on them
            assert not state.hasEnded()
            record = dict()
            record["gameCtor"] = state.getGameConstructorName()
            record["gameParams"] = state.getGameConstructorParams()
            record["knownResults"] = [result]
            record["generics"] = dict(iPolicy[1])
            record["policyIterated"] = iPolicy[0]
            record["uuid"] = str(uuid.uuid4())
            record["parent"] = prevStateUUID
            prevStateUUID = record["uuid"]
            record["policyUUID"] = policyUUID
            record["state"] = state.store()
            record["gamename"] = state.getGameName()
            record["creation"] = datetime.datetime.utcnow().timestamp()

            if self.featureProvider is not None:
                # theses only add the relevant state first, below they will be rewritten to the features from the feature providing network
                record["winFeatures"] = getNextState(trackList, ti, 0)
                record["winFeatures+1"] = getNextState(trackList, ti, 1)
                record["winFeatures+2"] = getNextState(trackList, ti, 2)
                record["winFeatures+3"] = getNextState(trackList, ti, 3)

                record["moveFeatures"] = getNextState(trackList, ti, 0)
                record["moveFeatures+1"] = getNextState(trackList, ti, 1)
                record["moveFeatures+2"] = getNextState(trackList, ti, 2)
                record["moveFeatures+3"] = getNextState(trackList, ti, 3)

            record["final"] = False

            # if the game is over after the current move is played, encode that as a uniform distribution over all moves
            reply = np.ones_like(record["policyIterated"])
            reply /= reply.shape[0]
            if (ti + 1) < len(trackList):
                # if however there is another move played, track the distribution for that move
                reply = trackList[ti+1][1][0]
            record["reply"] = reply

            reports.append(record)

        self.tracking[idx] = None

        if self.featureProvider is not None:
            fillRecordForFeatures(self.featureProvider, reports, self.device)

        if len(reports) > 0:
            reports[len(reports) - 1]["final"] = True
            self.gameReporter.reportGame(reports)

    def finalizeGames(self):
        """
        replace games that have been completed with new games, if desired by the impl.
        After this call all games in self.open should not be in a terminal state.
        The default impl just filters games out and replaces them with new games.
        """
        cdef int idx

        newList = []
        for idx, o in enumerate(self.open):
            if o.hasEnded():
                self.handleReportsFor(idx)
                newList.append(self.initialState)
            else:
                newList.append(o)

        self.open = newList



