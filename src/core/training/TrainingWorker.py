from core.training.StatesDownloader import StatesDownloader
import time
from utils.prints import logMsg
import abc

import sys
from utils.bsonHelp.bsonHelp import decodeFromBson, encodeToBson

import requests

class TrainingWindowManager(metaclass=abc.ABCMeta):
    """
    Responsible for deciding on the training window
    """

    @abc.abstractmethod
    def prepareWindow(self, downloader, currentIteration):
        """
        This function is allowed to use time.sleep internally and wait for the state of the StatesDownloader to change,
        as the StatesDownloader runs its own Thread, which updates the available data, even while this method blocks.

        The downloader is an active StatesDownloader object. It downloads new states automatically and makes them available to construct windows from.
        Relevant properties of the downloader are:
        downloader.numStatesAvailable how many states have been downloaded. More recent states get downloaded first.
        downloader.history history of states that have been generated. Orders by creation time in descending order.
            Each history object has properties: id, packageSize, worker, creation (unix ms number), iteration, network, they directly from from the /api/state/list webservice of the command server.

        To load a state package use downloader.openPackage(id), where id is taken from the downloadedStatesHistory objects.
        This yields a state-package (a list of dicts) as described by the (linear)selfplay worker. Unless you care about the specific content of the states you can just concat those lists to a single
        one and return that list.
        @return a list of state objects to be learned from.
        """

class TrainingWorker():

    def __init__(self, epochs, dataDir, policy, windowManager):
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
        self.windowManager = windowManager
        self.dataDir = dataDir
        self.downloader = StatesDownloader(self.dataDir)
        self.policy = policy

        hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for the trainer: --secret <server password>, --run <uuid> and --command <command server host>!")

        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.runId = sys.argv[sys.argv.index("--run")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]

    def getNetworkList(self):
        while True:
            try:
                networkList = requests.get(url=self.commandHost + "/api/networks/list/" + self.runId, headers={"secret": self.secret}).json()
                return networkList
            except Exception as error:
                logMsg("Could not get network list, hoping this is a temporary failure, will try again soon!")
                time.sleep(10)

    def uploadNetwork(self, newPolicyEncoded):
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

    def main(self):
        self.downloader.start()

        networkList = self.getNetworkList()
        if len(networkList) > 0:
            networkList.sort(lambda x: x["creation"], reversed=True)
            logMsg("Continue training of an existing network with id", networkList[0])
            networkId = networkList[0]["id"]

            response = requests.get(url=self.commandHost + "/api/networks/download/" + networkId, stream=True, headers={"secret": self.secret})
            response.raise_for_status()

            networkData = decodeFromBson(response.raw.data)

            self.policy.load(networkData)

            logMsg("Network %s loaded" % self.policy.getUUID())

        while True:
            window = self.windowManager.prepareWindow(self.downloader, len(networkList))

            startFitting = time.monotonic()
            self.policy.fit(window, self.epochs)
            endFitting = time.monotonic()

            logMsg("One iteration of training took %is" % int(endFitting - startFitting))

            newPolicyEncoded = encodeToBson(self.policy.store())

            self.uploadNetwork(newPolicyEncoded)

            networkList = self.getNetworkList()


            