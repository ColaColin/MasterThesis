from core.training.StatesDownloader import StatesDownloader
import time
from utils.prints import logMsg
import abc

from core.training.NetworkApi import NetworkApi

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

    def __init__(self, epochs, policy, windowManager):
        """
        Requires some configuration parameters to be present in the arguments to python
        --command <command server host>
        --secret <server api password>
        --run <run-uuid>

        Provides a main for the training worker. Old version, that waits until enough data for a training window is available
        and only then starts processing it. Probably no reason to use it anymore, use the StreamtrainingWorker instead,
        it speeds up network iteration times a lot and makes them more consistent in face of variation of the number
        of playing workers.
        Unlike playing workers there is only supposed to be one of these, running on a reliable and preferably very fast machine.
        Downloads states from the command server for the active run and decides on what kind of window of data to train
        @param epochs: Number of epochs to use to train a new network
        @param policy: The policy to use, should implement core.base.Policy
        @param windowManager: A TrainingWindowManager implementation.
        """
        self.epochs = epochs
        self.windowManager = windowManager
        self.policy = policy

        self.networks = NetworkApi()

        self.dataDir = "trainerData/" + self.networks.runId
        self.downloader = StatesDownloader(self.dataDir)

    def main(self):
        self.downloader.start()

        self.networks.loadNewestNetwork(self.policy)

        while True:
            window = self.windowManager.prepareWindow(self.downloader, len(networkList))

            startFitting = time.monotonic()
            self.policy.fit(window, self.epochs)
            endFitting = time.monotonic()

            logMsg("One iteration of training took %is" % int(endFitting - startFitting))

            self.networks.uploadNetwork(self.policy)

            networkList = self.networks.getNetworkList()


            