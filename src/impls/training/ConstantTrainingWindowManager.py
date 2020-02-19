import abc

from core.training.TrainingWorker import TrainingWindowManager
from utils.prints import logMsg
import time

class ConstantTrainingWindowManager(metaclass=abc.ABCMeta):
    """
    Configure a Training Window with a maximum size and a constant number of new states per network iteration.
    Will block until the right number of new states in the newest iteration is available before training the next network.
    Only use exactly that many new states from the current iteration.
    However since states are also generated while the network trains,
    some of the older states will also be replaced by newer states that were generated while the last network was trained.
    Try to keep training time short relative to how many new states are required for a new network.
    Worker time is wasted iff training takes long enough for the workers to replace more than the entire size of the window in the meantime.
    """

    def __init__(self, maxSize, nextIterationStatesCount):
        self.maxSize = maxSize
        self.nextIterationStatesCount = nextIterationStatesCount

    def countAvailableForIteration(self, downloader, iteration):
        available = 0
        hs = downloader.history.copy()
        for h in hs:
            if h["iteration"] == iteration:
                available += h["packageSize"]
        return available

    def getNextStatesForIteration(self, downloader, iteration):
        hs = downloader.history.copy()
        his = list(filter(lambda x: x["iteration"] == iteration, hs))
        his.sort(key = lambda x: x["creation"])
        num = 0
        activeList = []
        for hi in his:
            activeList.append(hi)
            num += hi["packageSize"]
            if num >= self.nextIterationStatesCount:
                break
        result = []
        for active in activeList:
            for state in downloader.openPackage(active["id"]):
                result.append(state)
        return result

    def getStatesBeforeIteration(self, downloader, iteration, num):
        hs = downloader.history.copy()
        count = 0
        activeList = []
        for h in hs:
            if h["iteration"] < iteration:
                activeList.append(h)
                count += h["packageSize"]
                if count >= num:
                    break
        result = []
        for active in activeList:
            for state in downloader.openPackage(active["id"]):
                result.append(state)
        return result

    def prepareWindow(self, downloader, currentIteration):
        logMsg("Need to prepare a window for iteration", currentIteration)

        windowSize = currentIteration * self.nextIterationStatesCount
        if windowSize > self.maxSize:
            windowSize = self.maxSize

        printed = False
        while self.countAvailableForIteration(downloader, currentIteration) < self.nextIterationStatesCount:
            if not printed:
                printed = True
                logMsg("Waiting for more data to train next network!")
            time.sleep(0.5)
        
        logMsg("Have enough data to train next network now!")

        newPolicyStates = self.getNextStatesForIteration(downloader, currentIteration)
        olderPolicyStates = self.getStatesBeforeIteration(downloader, currentIteration, self.maxSize - self.nextIterationStatesCount)

        trainingWindow = newPolicyStates + olderPolicyStates
        logMsg("Using a window of %i states with %i states representing the new policy" % (len(trainingWindow), len(newPolicyStates)))
        return trainingWindow

        
        

            