"""
Supervised training script to establish the maximum possible playing strength with the given network.
"""

import setproctitle
from utils.prints import logMsg, setLoggingEnabled
from core.solved.PolicyTester import loadExamples2, DatasetPolicyTester2, SolverBatchedPolicyPlayer, PolicyPlayer
from impls.selfplay.movedeciders import TemperatureMoveDecider
import os
import random
import numpy as np
import math

# Training set size | moves % | wins % 
# ------------------+---------+---------
#             10000 |  65.28  |  37.04  
#             20000 |  71.84  |  38.08  
#             30000 |  74.02  |  40.99  
#             40000 |  76.50  |  39.39   <<< best

class SupervisedNetworkTrainer():
    def __init__(self, datasetFile, initialGame, policy, windowSizeSplits, trainingRuns, workingDirectory, testSamples, validationSamples, batchSize, lrStart, lrPatience):
        logMsg("Starting to initialize supervised training")

        self.examples = loadExamples2(initialGame, datasetFile)
        random.seed(42)
        random.shuffle(self.examples)

        self.records = []
        for lex in self.examples:
            record = dict()
            record["state"] = lex[0].store()

            moveOutput = np.zeros(lex[0].getMoveCount(), dtype=np.float32)
            for m in lex[2]:
                moveOutput[m] += 1
            moveOutput /= np.sum(moveOutput)
            record["policyIterated"] = moveOutput

            assert lex[0].getPlayerCount() == 2, "The whole solved dataset thing is not really meant for more than 2 players right now. Would need to rethink the file format a bit"
            gresult = []
            if lex[3] == 0:
                gresult = [0]
            elif lex[3] == -1:
                gresult = [(lex[0].getPlayerOnTurnNumber() + 1) % lex[0].getPlayerCount()]
            elif lex[3] == 1:
                gresult = [lex[0].getPlayerOnTurnNumber()]
            else:
                assert False, "Expected game result from database file to be in [-1, 0, 1]"
            
            record["knownResults"] = gresult

            self.records.append(record)

        self.initialGame = initialGame
        self.policy = policy
        self.windowSizeSplits = windowSizeSplits
        self.trainingRuns = trainingRuns
        self.workingDirectory = workingDirectory
        self.testSamples = testSamples
        self.validationSamples = validationSamples
        self.trainingSamples = len(self.records) - self.testSamples - self.validationSamples
        self.batchSize = batchSize
        self.lrStart = lrStart
        self.lrPatience = lrPatience
        self.trainingResults = []

        logMsg("Initialization completed")


    def doRun(self, workDir, trainingSet, validationSet, testSet):
        logMsg("Supervised training starts for %i training examples, validating on %i examples and testing on %i examples." % (len(trainingSet), len(validationSet), len(testSet)))
        os.makedirs(workDir, exist_ok=True)

        logPath = os.path.join(workDir, "training.log")

        with open(logPath, "a+") as logf:
            self.runTraining(workDir, logf, trainingSet, validationSet)

            moveAcc, winAcc = self.evaluateNetwork(testSet)

            testPrnt = "Network achieved a test accuracy of %.2f%% moves and %.2f%% wins\n\n" % (moveAcc, winAcc)
            logMsg(testPrnt)
            logf.write(testPrnt + "\n")

        return moveAcc, winAcc

    def runTraining(self, workDir, logFileHandle, trainingSet, validationSet):
        self.policy.reset()

        def logTxt(txt):
            logMsg(txt)
            logFileHandle.write(txt + "\n")

        epoch = 0
        lastImprovement = 0
        bestNetworkPath = None
        lastReduction = 0
        currentValidationScore = None
        curLr = self.lrStart

        numBatches = math.ceil(len(trainingSet) / self.batchSize)

        while epoch - lastImprovement < self.lrPatience * 1.5:
            if epoch - max(lastImprovement, lastReduction) >= self.lrPatience:
                lastReduction = epoch
                curLr /= 10
                logTxt("!!!!!\n[Epoch %03d] drops the LR" % epoch)

            moveLosses = []
            winLosses = []

            for batchIndex in range(numBatches):
                batchData = trainingSet[batchIndex * self.batchSize : (batchIndex + 1) * self.batchSize]
                mls, wls, rls = self.policy.fit(self.policy.quickPrepare(batchData), forceLr = curLr)
                assert rls is None, "reply prediction is not supported in supervised training!"
                moveLosses += mls
                winLosses += wls

            moveAccuarcy, winAccuracy = self.evaluateNetwork(validationSet)

            logTxt("[Epoch %03d] with LR %0.4f, loss: moves %.5f, wins: %.5f; val: moves %.2f%%, wins: %.2f%%" % (epoch, curLr, np.mean(moveLosses), np.mean(winLosses),  moveAccuarcy, winAccuracy))
            validationScore = (moveAccuarcy + winAccuracy) / 2

            epochPath = os.path.join(workDir, "epoch_" + str(epoch) + ".npy")
            np.save(epochPath, self.policy.store(), allow_pickle=False)

            if currentValidationScore is None:
                currentValidationScore = validationScore
                bestNetworkPath = epochPath

            elif currentValidationScore < validationScore:
                logTxt("!!!!!\n[Epoch %03d] Improved validation result from %.2f%% to %.2f%%!" % (epoch, currentValidationScore, validationScore))
                currentValidationScore = validationScore
                bestNetworkPath = epochPath
                lastImprovement = epoch

            epoch += 1

        logMsg("Training ended, best network was %s, loading that network" % bestNetworkPath)
        self.policy.load(np.load(bestNetworkPath))

    def evaluateNetwork(self, onDataSet):
        setLoggingEnabled(False)
        tester = DatasetPolicyTester2(PolicyPlayer(self.policy, None, TemperatureMoveDecider(-1)), None, self.initialGame, onDataSet)
        result = tester.main()
        setLoggingEnabled(True)
        return result

    def main(self):
        setproctitle.setproctitle("x0_supervised")

        windowStep = self.trainingSamples // self.windowSizeSplits
        for i in range(1, self.windowSizeSplits+1):
            trainSetSize = i * windowStep
            workDir = os.path.join(self.workingDirectory, "supervised_trainsetsize_" + str(i))
            mas = []
            was = []
            for k in range(self.trainingRuns):
                kdir = os.path.join(workDir, "run_" + str(k))
                trainSet = self.records[:trainSetSize]
                validationSet = self.examples[trainSetSize:trainSetSize + self.validationSamples]
                assert (trainSetSize + self.validationSamples - 1) < len(self.examples)-self.testSamples
                testSet = self.examples[-self.testSamples:]
                moveAccuracy, winAccuracy = self.doRun(kdir, trainSet, validationSet, testSet)
                mas.append(moveAccuracy)
                was.append(winAccuracy)

            self.trainingResults.append((trainSetSize, np.mean(mas), np.mean(was)))

        bestIndex = -1
        bestScore = 0
        for i in range(len(self.trainingResults)):
            tr = self.trainingResults[i]
            score = tr[1] + tr[2]
            if score > bestScore:
                bestScore = score
                bestIndex = i

        finalResultTxt =  "\nTraining set size | moves % | wins % \n"
        finalResultTxt += "------------------+---------+---------\n"
        for i in range(len(self.trainingResults)):
            finalResultTxt += "%17d |  %.2f  |  %.2f  " % self.trainingResults[i]
            if i == bestIndex:
                finalResultTxt += " <<< best\n"
            else:
                finalResultTxt += "\n"

        logMsg(finalResultTxt)