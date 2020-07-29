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

import torch
from torch.autograd import Variable
import sys
from impls.polices.pytorch.policy import unpackTorchNetwork

from utils.bsonHelp.bsonHelp import decodeFromBson

from impls.selfplay.LinearSelfPlay import fillRecordForFeatures

# Training set size | moves % | wins % 
# ------------------+---------+---------
#             10000 |  65.28  |  37.04  
#             20000 |  71.84  |  38.08  
#             30000 |  74.02  |  40.99  
#             40000 |  76.50  |  39.39   <<< best


# no extra features
# [2020-07-16T18:20:18.087355+02:00] 
# Training set size | moves % | wins % 
# ------------------+---------+---------
#              8000 |  75.76  |  70.72   <<< best

#   useWinFeatures: 0
#   useMoveFeatures: 0
#   featuresWeight: 0.01
# [2020-07-16T18:34:54.154848+02:00] 
# Training set size | moves % | wins % 
# ------------------+---------+---------
#              8000 |  75.12  |  70.48   <<< best

#   useWinFeatures: 0
#   useMoveFeatures: 0
#   featuresWeight: 0.001
# [2020-07-16T18:43:11.856511+02:00] 
# Training set size | moves % | wins % 
# ------------------+---------+---------
#              8000 |  75.20  |  70.90   <<< best


#   useWinFeatures: -1
#   useMoveFeatures: -1
#   featuresWeight: 0.001
# [2020-07-16T18:59:10.090574+02:00] 
# Training set size | moves %         |  wins % 
# ------------------+---------+---------
#              8000 |  76.16 +/- 1.50 |  71.06 +/- 1.37  <<< best


#   useWinFeatures: 3
#   useMoveFeatures: 3
#   featuresWeight: 0.0001
# [2020-07-16T19:10:09.426264+02:00] 
# Training set size | moves %         |  wins % 
# ------------------+---------+---------
#              8000 |  75.76 +/- 1.11 |  70.58 +/- 0.55  <<< best


#   useWinFeatures: 3
#   useMoveFeatures: 3
#   featuresWeight: 0.1
# [2020-07-16T19:23:39.398682+02:00] 
# Training set size | moves %         |  wins % 
# ------------------+---------+---------
#              8000 |  77.10 +/- 1.52 |  70.58 +/- 0.32  <<< best


#   useWinFeatures: 3
#   useMoveFeatures: 3
#   featuresWeight: 1
# [2020-07-16T19:34:06.262367+02:00] 
# Training set size | moves %         |  wins % 
# ------------------+---------+---------
#              8000 |  78.62 +/- 0.83 |  70.50 +/- 0.78  <<< best


#   useWinFeatures: 3
#   useMoveFeatures: 3
#   featuresWeight: 10
# [2020-07-16T19:44:14.155687+02:00] 
# Training set size | moves %         |  wins % 
# ------------------+---------+---------
#              8000 |  78.62 +/- 0.73 |  70.04 +/- 0.84  <<< best


#   useWinFeatures: 0
#   useMoveFeatures: 0
#   featuresWeight: 1
# [2020-07-16T19:52:57.325871+02:00] 
# Training set size | moves %         |  wins % 
# ------------------+---------+---------
#              8000 |  80.38 +/- 0.71 |  71.80 +/- 0.53  <<< best




class SupervisedNetworkTrainer():
    def __init__(self, datasetFile, initialGame, policy, windowSizeSplits, trainingRuns, workingDirectory, testSamples, validationSamples, batchSize, lrStart, lrPatience, featureProvider=None, featureNetwork=None):
        logMsg("Starting to initialize supervised training")

        self.featureProvider = featureProvider
        self.featureNetwork = featureNetwork

        if self.featureProvider is not None:
            logMsg("Using feature provider network!")
            if self.featureNetwork is not None:
                with open(self.featureNetwork, "rb") as f:
                    networkData = decodeFromBson(f.read())
                    uuid, modelDict, netConfig = unpackTorchNetwork(networkData)
                    self.featureProvider.load_state_dict(modelDict)
                    logMsg("Loaded feature network %s" % uuid)

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

        self.examples = loadExamples2(initialGame, datasetFile)
        random.seed(42)
        random.shuffle(self.examples)

        sawFutures = False

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

            if len(lex) > 4 and self.featureProvider is not None:
                if not sawFutures:
                    sawFutures = True
                    logMsg("Dataset contains future positions to create features from!")

                record["winFeatures"] = lex[0]
                record["winFeatures+1"] = lex[-1][0]
                record["winFeatures+2"] = lex[-1][1]
                record["winFeatures+3"] = lex[-1][2]

                record["moveFeatures"] = lex[0]
                record["moveFeatures+1"] = lex[-1][0]
                record["moveFeatures+2"] = lex[-1][1]
                record["moveFeatures+3"] = lex[-1][2]

            self.records.append(record)

        if sawFutures:
            logMsg("Creating extra features!")
            bsize = 1024
            for ix in range(0, len(self.records), bsize):
                batch = self.records[ix:(ix+bsize)]
                fillRecordForFeatures(self.featureProvider, batch, self.device)
            logMsg("Extra features created!")

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
            wfLosses = []
            mfLosses = []

            for batchIndex in range(numBatches):
                batchData = trainingSet[batchIndex * self.batchSize : (batchIndex + 1) * self.batchSize]
                mls, wls, rls, wfl, mfl = self.policy.fit(self.policy.quickPrepare(batchData), forceLr = curLr)
                assert rls is None or len(rls) == 0, "reply prediction is not supported in supervised training!"
                moveLosses += mls
                winLosses += wls
                wfLosses += wfl
                mfLosses += mfl

            moveAccuarcy, winAccuracy = self.evaluateNetwork(validationSet)

            extrastr = ""
            if len(wfLosses) > 0:
                extrastr += ", win features %.5f," % np.mean(wfLosses)
            if len(mfLosses) > 0:
                extrastr += " move features %.5f" % np.mean(mfLosses)

            logTxt(("[Epoch %03d] with LR %0.4f, loss: moves %.5f, wins: %.5f; val: moves %.2f%%, wins: %.2f%%" % (epoch, curLr, np.mean(moveLosses), np.mean(winLosses),  moveAccuarcy, winAccuracy)) + extrastr)
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

            self.trainingResults.append((trainSetSize, np.mean(mas), np.std(mas), np.mean(was), np.std(was)))

        bestIndex = -1
        bestScore = 0
        for i in range(len(self.trainingResults)):
            tr = self.trainingResults[i]
            score = tr[1] + tr[2]
            if score > bestScore:
                bestScore = score
                bestIndex = i

        finalResultTxt =  "\nTraining set size | moves %         |  wins % \n"
        finalResultTxt += "------------------+---------+---------\n"
        for i in range(len(self.trainingResults)):
            finalResultTxt += "%17d |  %.2f +/- %.2f |  %.2f +/- %.2f " % self.trainingResults[i]
            if i == bestIndex:
                finalResultTxt += " <<< best\n"
            else:
                finalResultTxt += "\n"

        logMsg(finalResultTxt)