from core.training.StatesDownloader import StatesDownloader
import time
from utils.prints import logMsg

import sys
from utils.bsonHelp.bsonHelp import decodeFromBson, encodeToBson

import requests

class TrainingWorker():

    def __init__(self, epochs, windowSize, dataDir, policy, minWindowsSize = None):
        """
        Requires some configuration parameters to be present in the arguments to python
        --command <command server host>
        --secret <server api password>
        --run <run-uuid>

        Provides a main for the training worker. Currently not used with the core.mains.distributed main, but instead started
        with a local config using the normal core.mains.main.
        Unlike playing workers there is only supposed to be one of these, running on a reliable and preferably very fast machine.
        Downloads states from the command server for the active run and decides on what kind of window of data to train
        @param epochs: Number of epochs to use to train a new network
        @param minWindowsSize: Wait for this many states to be available, before the first network is trained. If None given, use windowSize
        @param windowSize: Number of recent game states to be included in training the next network
        @param dataDir: Directory to use 
        @param policy: The policy to use, should implement core.base.Policy
        """
        self.epochs = epochs
        self.minWindowsSize = minWindowsSize
        if self.minWindowsSize is None:
            self.minWindowsSize = windowSize
        self.windowSize = windowSize
        self.dataDir = dataDir
        self.downloader = StatesDownloader(self.dataDir)
        self.policy = policy
        self.lastTrainingIds = set()
        self.firstWindow = True

        hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for the trainer: --secret <server password>, --run <uuid> and --command <command server host>!")

        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.runId = sys.argv[sys.argv.index("--run")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]


    def waitForData(self):
        while self.downloader.numStatesAvailable < self.minWindowsSize:
            logMsg("Waiting for initial training window, have %i states available, need %i" % (self.downloader.numStatesAvailable, self.windowSize))
            time.sleep(20)

    def loadNextWindow(self):
        numStates = 0
        activeList = []
        for h in self.downloader.downloadedStatesHistory:
            activeList.append(h)
            numStates += h["packageSize"]
            if numStates > self.windowSize:
                break
        loaded = []
        newStatesForWindow = 0
        oldStatesForWindow = 0
        for active in activeList:
            stateList = self.downloader.openPackage(active["id"])
            if active["id"] in self.lastTrainingIds:
                oldStatesForWindow += active["packageSize"]
            else:
                newStatesForWindow += active["packageSize"]

            for state in stateList:
                loaded.append(state)

        if oldStatesForWindow == 0 and not self.firstWindow:
            logMsg("\n!\n!!\n!!!\n==================\nWARNING: Trainer is too slow, not all generated data is used for training!\n==================\n!!!\n!!\n!\n")

        self.lastTrainingIds = set()
        for active in activeList:
            self.lastTrainingIds.add(active["id"])

        logMsg("Loaded new window to train on, reused %i states, got %i new states!" % (oldStatesForWindow, newStatesForWindow))

        self.firstWindow = False
        return loaded
        

    def main(self):
        self.downloader.start()

        networkList = requests.get(url=self.commandHost + "/api/networks/list/" + self.runId, headers={"secret": self.secret}).json()
        if len(networkList) > 0:
            networkList.sort(lambda x: x["creation"], reversed=True)
            logMsg("Continue training of an existing network with id", networkList[0])
            networkId = networkList[0]["id"]

            response = requests.get(url=self.commandHost + "/api/networks/download/" + networkId, stream=True, headers={"secret": self.secret})
            response.raise_for_status()

            networkData = decodeFromBson(response.raw.data)

            self.policy.load(networkData)

            logMsg("Network loaded")

        self.waitForData()

        while True:
            window = self.loadNextWindow()

            startFitting = time.monotonic()
            self.policy.fit(window, self.epochs)
            endFitting = time.monotonic()

            logMsg("One iteration of training took %is" % int(endFitting - startFitting))

            newPolicyEncoded = encodeToBson(self.policy.store())

            while True:
                try:
                    response = requests.post(url=self.commandHost + "/api/networks/" + self.runId + "/" + self.policy.getUUID(), data=newPolicyEncoded,
                        headers={"secret": self.secret})
                    response.raise_for_status()

                    logMsg("Network uploaded successfully!")

                    break
                except Exception as error:
                    logMsg("Could not upload network, hoping this is a temporary failure, will try again soon!")
                    time.sleep(60)


            