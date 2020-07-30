# cython: profile=False

import os
import abc

import numpy as np
cimport numpy as np

import torch
import torch.cuda
import torch.nn as nn
from torch.autograd import Variable

import uuid

from core.base.Policy import Policy, ExamplePrepareWorker

from utils.encoding import stringToBytes, bytesToString

import pytorch_model_summary as pms

import io

import random

import math

import time

from utils.misc import constructor_for_class_name, IterationCalculatedValue

from utils.prints import logMsg

import sys

from torch.nn.utils.clip_grad import clip_grad_norm_

# helper function to unpack data that is used to encode the policy as bytes.
def unpackTorchNetwork(packed):
    if packed[0] == 255:
        numBlocks = packed[1]
        numFilters = (packed[2] << 8) | packed[3]
        headKernel = packed[4]
        numHeadFilters = (packed[5] << 8) | packed[6]
        mode = "plain" if packed[7] == 1 else "sq"
        
        ublen = packed[8]
        uuidBytes = packed[9:ublen+9]
        modelBuffer = io.BytesIO(bytes(packed[ublen+9:]))

        return bytesToString(uuidBytes), torch.load(modelBuffer, map_location=torch.device('cpu')),\
            [numBlocks, numFilters, headKernel, numHeadFilters, mode]
    else:
        ublen = packed[0]
        uuidBytes = packed[1:ublen+1]
        modelBuffer = io.BytesIO(bytes(packed[ublen+1:]))

        return bytesToString(uuidBytes), torch.load(modelBuffer, map_location=torch.device('cpu')), []

# SELayer was taken from https://github.com/moskomule/senet.pytorch/blob/9d279eccb5a0ca6cb09ad1053b5f971656b801de/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResBlock(nn.Module):
    def __init__(self, features):
        super(SEResBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)
        self.act = nn.ReLU(inplace=True)
        self.se = SELayer(features)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se(out)

        out += residual
        
        out = self.act(out)
        
        return out

class ResBlock(nn.Module):
    def __init__(self, features):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        
        out = self.act(out)
        
        return out

class ResCNN(nn.Module):
    def __init__(self, inWidth, inHeight, inDepth, baseKernelSize, baseFeatures, features,\
            blocks, moveSize, winSize, extraHeadFilters, mode="plain", predictReply=False, outputExtra=None):
        super(ResCNN, self).__init__()

        paddingSize = 1 if baseKernelSize > 2 else 0

        self.predictReply = predictReply

        self.outputExtra = outputExtra

        self.baseConv = nn.Conv2d(inDepth, baseFeatures, baseKernelSize, padding=paddingSize)
        self.baseBn = nn.BatchNorm2d(baseFeatures)
        self.act = nn.ReLU(inplace=True)
        
        if (baseFeatures != features and blocks > 0):
            self.matchConv = nn.Conv2d(baseFeatures, features, 1)
        else:
            self.matchConv = None
        
        if mode == "plain":
            logMsg("Using plain ResNet!")
            rblock = ResBlock
        elif mode == "sq":
            logMsg("Using squeeze-excite ResNet!")
            rblock = SEResBlock

        blockList = []
        for _ in range(blocks):
            blockList.append(rblock(features))
        self.resBlocks = nn.Sequential(*blockList)

        self.extraHeadFilters = extraHeadFilters;
        if self.extraHeadFilters is not None:
            self.moveHeadConv = nn.Conv2d(features, self.extraHeadFilters, 3, padding=1)
            self.valueHeadConv = nn.Conv2d(features, self.extraHeadFilters, 3, padding=1)
            hiddens = self.extraHeadFilters * (inWidth - (baseKernelSize - (1 + paddingSize * 2))) * (inHeight - (baseKernelSize - (1 + paddingSize * 2)))
        else:
            hiddens = features * (inWidth - (baseKernelSize - (1 + paddingSize * 2))) * (inHeight - (baseKernelSize - (1 + paddingSize * 2)))

        self.moveHead = nn.Linear(hiddens, moveSize)

        if self.outputExtra == "extraConvOutput":
            self.extraHead = nn.Conv2d(features, 8, 3, padding=1)

        if self.predictReply:
            self.replyHead = nn.Linear(hiddens, moveSize)

        assert winSize > 0

        self.winHead = nn.Linear(hiddens, winSize)
            
        self.lsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.act(self.baseBn(self.baseConv(x)))
        
        if (self.matchConv != None):
            x = self.matchConv(x)
        
        x = self.resBlocks(x)
        
        if self.extraHeadFilters is None:
            x = x.view(x.size(0), -1)

            moveP = self.lsoftmax(self.moveHead(x))

            if self.predictReply:
                replyP = self.lsoftmax(self.replyHead(x))
            else:
                replyP = None

            winP = self.lsoftmax(self.winHead(x))
            return moveP, winP, replyP
        else:
            preHead = x
            moveX = self.moveHeadConv(x)
            winX = self.valueHeadConv(x)

            headMove = moveX
            headWin = winX

            moveX = moveX.view(moveX.size(0), -1)
            winX = winX.view(winX.size(0), -1)

            moveP = self.lsoftmax(self.moveHead(moveX))

            if self.predictReply:
                replyP = self.lsoftmax(self.replyHead(moveX))
            else:
                replyP = None

            winP = self.lsoftmax(self.winHead(winX))

            if self.outputExtra is None:
                return moveP, winP, replyP
            elif self.outputExtra == "winhead":
                return moveP, winP, replyP, headWin
            elif self.outputExtra == "movehead":
                return moveP, winP, replyP, headMove
            elif self.outputExtra == "bothhead":
                return moveP, winP, replyP, headWin, headMove
            elif self.outputExtra == "resblock":
                return moveP, winP, replyP, preHead, preHead
            elif self.outputExtra == "extraConvOutput":
                eOut = self.extraHead(preHead)
                return moveP, winP, replyP, eOut, eOut
            else:
                assert False, "Unknown outputExtra: " + self.outputExtra
        
def gameResultsToAbsoluteWinTensor(wins, playerCount):
    result = np.zeros(playerCount, dtype=np.float32)
    for win in wins:
        result[win] += 1
    result /= np.sum(result)
    return result

class PytorchExamplePrepareWorker(ExamplePrepareWorker, metaclass=abc.ABCMeta):
    def __init__(self, device, gameCtor, gameParams, gameDims, targetReply, useWinFeatures = -1, useMoveFeatures = -1):
        self.device = device
        self.gameCtor = gameCtor
        self.gameParams = gameParams
        self.protoState = None
        self.gameDims = gameDims
        self.targetReply = targetReply

        self.useWinFeatures = useWinFeatures
        self.useMoveFeatures = useMoveFeatures

    def createFeatureTensor(self, frame, featureName):
        wtf = frame[featureName]
        ftensor = torch.zeros((1, len(wtf)))
        cdef float [:, :] ftensornp = ftensor.numpy()
        cdef int idx

        for idx, f in enumerate(wtf):
            ftensornp[0, idx] = f

        return ftensor

    def prepareExample(self, frame):
        if self.protoState is None:
            ctor = constructor_for_class_name(self.gameCtor)
            self.protoState = ctor(**self.gameParams)

        inputArray = torch.zeros((1, ) + self.gameDims)
        outputMoves = torch.zeros((1, self.protoState.getMoveCount()))
        outputWins = torch.zeros((1, (self.protoState.getPlayerCount() + 1)))

        game = self.protoState.load(frame["state"])

        cdef float [:, :, :, :] inputTensor = inputArray.numpy()
        cdef float [:, :] moveTensor = outputMoves.numpy()
        cdef float [:, :] winTensor = outputWins.numpy()

        newPolicy = frame["policyIterated"]

        game.encodeIntoTensor(inputTensor, 0, False)

        cdef int idx, pid, mappedIndex

        for idx, p in enumerate(newPolicy):
            moveTensor[0, idx] = p

        absoluteWinners = gameResultsToAbsoluteWinTensor(frame["knownResults"], game.getPlayerCount() + 1)

        winTensor[0, 0] = absoluteWinners[0]
        for pid in range(1, game.getPlayerCount() + 1):
            mappedIndex = game.mapPlayerNumberToTurnRelative(pid) + 1
            winTensor[0, mappedIndex] = absoluteWinners[pid]

        cdef float [:, :] replyTensor;

        result = []

        if self.targetReply:
            outputReply = torch.zeros((1, self.protoState.getMoveCount()))
            replyTensor = outputReply.numpy()
            for idx, p in enumerate(frame["reply"]):
                replyTensor[0, idx] = p
            result += [inputArray, outputMoves, outputWins, outputReply, hash(game)]
        else:
            result += [inputArray, outputMoves, outputWins, hash(game)]

        if self.useWinFeatures != -1 or self.useMoveFeatures != -1:
            featuresDicts = dict()
            result.append(featuresDicts)

            featureNames = ["winFeatures", "winFeatures+1", "winFeatures+2", "winFeatures+3",\
                "moveFeatures", "moveFeatures+1", "moveFeatures+2", "moveFeatures+3"]

            for fname in featureNames:
                if fname in frame:
                    featuresDicts[fname] = self.createFeatureTensor(frame, fname)
        
        return result

    def getHashForExample(self, example):
        if self.targetReply:
            return example[4]
        else:
            return example[3]

    def areExamplesEqual(self, exampleA, exampleB):
        return torch.equal(exampleA[0], exampleB[0])

    def mergeInto(self, target, source, sourceWeight):
        targetWeight = 1 - sourceWeight

        target[1] *= targetWeight
        target[1] += sourceWeight * source[1]

        target[2] *= targetWeight
        target[2] += sourceWeight * source[2]

        if self.targetReply:
            target[3] *= targetWeight
            target[3] += sourceWeight * source[3]

        if self.useWinFeatures != -1 or self.useMoveFeatures != -1:
            ftdict = target[len(target)-1]
            fsdict = source[len(source)-1]
            for fname in dict.keys(ftdict):
                ftdict[fname] *= targetWeight
                ftdict[fname] += sourceWeight * fsdict[fname]

    def packageExamplesBatch(self, examples):
        # format is (inputs, movesOut, winsOut, examplesCount)[]
        random.shuffle(examples)
        inputs = torch.cat(list(map(lambda x: x[0], examples))).to(self.device)
        movesOut = torch.cat(list(map(lambda x: x[1], examples))).to(self.device)
        winsOut = torch.cat(list(map(lambda x: x[2], examples))).to(self.device)

        result = []

        if self.targetReply:
            replyOut = torch.cat(list(map(lambda x: x[3], examples))).to(self.device)
            result += [inputs, movesOut, winsOut, replyOut, len(examples)]
        else:
            result += [inputs, movesOut, winsOut, len(examples)]

        if self.useWinFeatures != -1 or self.useMoveFeatures != -1:
            fidx = len(examples[0]) - 1
            winFeaturesOut = [
                torch.cat(list(map(lambda x: x[fidx]["winFeatures"], examples))).to(self.device),
                torch.cat(list(map(lambda x: x[fidx]["winFeatures+1"], examples))).to(self.device),
                torch.cat(list(map(lambda x: x[fidx]["winFeatures+2"], examples))).to(self.device),
                torch.cat(list(map(lambda x: x[fidx]["winFeatures+3"], examples))).to(self.device)
            ]
            moveFeaturesOut = [
                torch.cat(list(map(lambda x: x[fidx]["moveFeatures"], examples))).to(self.device),
                torch.cat(list(map(lambda x: x[fidx]["moveFeatures+1"], examples))).to(self.device),
                torch.cat(list(map(lambda x: x[fidx]["moveFeatures+2"], examples))).to(self.device),
                torch.cat(list(map(lambda x: x[fidx]["moveFeatures+3"], examples))).to(self.device)
            ]

            result += [winFeaturesOut, moveFeaturesOut]

        return result
    

class LrStepSchedule(IterationCalculatedValue, metaclass=abc.ABCMeta):
    def __init__(self, startValue, stepEvery, stepMultiplier, minValue):
        self.startValue = startValue
        self.stepEvery = stepEvery
        self.stepMultiplier = stepMultiplier
        self.minValue = minValue
        self.lastLr = 0
    
    def getValue(self, iteration, iterationProgress):
        steps = iteration // self.stepEvery
        val = self.startValue

        for _ in range(steps):
            val *= self.stepMultiplier
        
        if val < self.minValue:
            val = self.minValue

        if self.lastLr == 0 or abs(val - self.lastLr) / self.lastLr > 0.1:
            logMsg("Setting new LR", val)
            self.lastLr = val

        return val

class OneCycleSchedule(IterationCalculatedValue, metaclass=abc.ABCMeta):
    def __init__(self, peak, end, baseVal, peakVal, endVal, dbgName="one_cycle", targetIteration=0, targetReductionFactor=1):
        self.peak = peak
        self.end = end
        self.baseVal = baseVal
        self.peakVal = peakVal
        self.endVal = endVal
        self.lastPhase = "end" # inc -> dec -> end
        self.name = dbgName
        self.targetIteration = targetIteration
        self.targetReductionFactor = targetReductionFactor

    def getValue(self, iteration, iterationProgress):
        currentPhase = "inc"
        if iterationProgress >= self.peak:
            currentPhase = "dec"
        if iterationProgress >= self.end:
            currentPhase = "end"

        if currentPhase != self.lastPhase:
            logMsg("OneCycle parameter %s enters new phase: %s" % (self.name, currentPhase))
            self.lastPhase = currentPhase

        if currentPhase == "inc":
            progress = iterationProgress / self.peak
            result = self.baseVal + progress * (self.peakVal - self.baseVal)
        elif currentPhase == "dec":
            progress = (iterationProgress - self.peak) / (self.end - self.peak)
            result = self.peakVal + progress * (self.baseVal - self.peakVal)
        elif currentPhase == "end":
            progress = (iterationProgress - self.end) / (1 - self.end)
            result = self.baseVal + progress * (self.endVal - self.baseVal)
        else:
            assert False, ("unknown currentPhase value: " + currentPhase)

        if (iteration - 1) >= self.targetIteration:
            result *= self.targetReductionFactor
        else:
            reduceDiff = 1 - self.targetReductionFactor
            targetFrac = ((iteration - 1) / self.targetIteration) * reduceDiff
            rFactor = 1 - targetFrac
            result *= rFactor

        return result

class PytorchPolicy(Policy, metaclass=abc.ABCMeta):
    """
    A policy that uses Pytorch to implement a ResNet-tower similar to the one used by the original AlphaZero implementation.
    different network variants can be used with the network parameter:
    plain: "classical" AlphaZero resnet
    sq: AlphaZero, but with Squeeze Excite elements
    useWinFeatures/useMoveFeatures: -1 to deactivate, 0 to only current turn, 1 to use next turn, 2 to use two next turns, 3 to use 3 next turns. Not above 3.
    featuresWeight: training weight for the loss created from win/move features.
    """

    def __init__(self, batchSize, blocks, filters, headKernel, headFilters, protoState, device, optimizerName, \
            optimizerArgs = None, extraHeadFilters = None, silent = True, lrDecider = None, gradClipValue = None, valueLossWeight = 1, momentumDecider = None, \
            networkMode="plain", replyWeight = 0,\
            useWinFeatures = -1, useMoveFeatures = -1, featuresWeight = 0.01,\
            outputExtra="bothhead"):
        self.batchSize = batchSize
        
        self.useWinFeatures = useWinFeatures
        self.useMoveFeatures = useMoveFeatures
        self.featuresWeight = featuresWeight

        self.outputExtra = outputExtra

        logMsg("extra features", self.useWinFeatures, self.useMoveFeatures, self.featuresWeight, self.outputExtra)

        if torch.cuda.is_available():
            gpuCount = torch.cuda.device_count()

            if "--windex" in sys.argv and gpuCount > 1 and device == "cuda":
                windex = int(sys.argv[sys.argv.index("--windex") + 1])
                gpuIndex = windex % gpuCount
                device = "cuda:" + str(gpuIndex)
                logMsg("Found multiple gpus with set windex, extended cuda device to %s" % device)

            self.device = torch.device(device)
        else:
            logMsg("No GPU is available, falling back to cpu!")
            self.device = torch.device("cpu")
        
        self.networkMode = networkMode

        self.extraHeadFilters = extraHeadFilters
        self.uuid = str(uuid.uuid4())
        self.gameDims = protoState.getDataShape()
        self.protoState = protoState

        self.optimizerName = optimizerName
        self.optimizerArgs = optimizerArgs

        if self.optimizerArgs is None:
            self.optimizerArgs = dict()

        self.blocks = blocks
        self.filters = filters
        self.headKernel = headKernel
        self.headFilters = headFilters
        assert len(self.gameDims) == 3, "ResNet requires a 3D data shape!"
        self.tensorCacheExists = False

        self.silent = silent

        self.gradClipValue = gradClipValue

        self.valueLossWeight = valueLossWeight

        self.replyWeight = replyWeight

        self.lrDecider = lrDecider
        self.momentumDecider = momentumDecider

        self.packer = PytorchExamplePrepareWorker(self.device, self.protoState.getGameConstructorName(), self.protoState.getGameConstructorParams(),\
            self.gameDims, self.replyWeight > 0, self.useWinFeatures, self.useMoveFeatures)

        self.initNetwork()

    def initNetwork(self):
        self.net = ResCNN(self.gameDims[1], self.gameDims[2], self.gameDims[0],\
            self.headKernel, self.headFilters, self.filters, self.blocks,\
            self.protoState.getMoveCount(), self.protoState.getPlayerCount() + 1, self.extraHeadFilters,\
            mode=self.networkMode, predictReply=self.replyWeight > 0, outputExtra=self.outputExtra)
        self.net = self.net.to(self.device)

        # can't use mlconfig, as mlconfig has no access to self.net.parameters :(
        octor = constructor_for_class_name(self.optimizerName)
        self.optimizer = octor(self.net.parameters(), **self.optimizerArgs)
        self.net.train(False)

        logMsg("\nCreated a PytorchPolicy using device " + str(self.device) + "\n", pms.summary(self.net, torch.zeros((1, ) + self.gameDims, device = self.device)))
        if self.replyWeight > 0:
            logMsg("PytorchPolicy will predict the next turns move policy with a weight of %.2f" % self.replyWeight)
        else:
            logMsg("Prediction of the next turns move policy is disabled")

    def prepareForwardData(self, batch, int thisBatchSize):
        if not self.tensorCacheExists:
            prep = torch.zeros((self.batchSize, ) + self.gameDims)
            if torch.cuda.is_available():
                prep = prep.pin_memory()
            self.tensorCacheCPU = prep
            self.forwardInputGPU = Variable(torch.zeros((self.batchSize, ) + self.gameDims), requires_grad=False).to(self.device)
            self.tensorCacheExists = True

        cdef float [:, :, :, :] npTensor = self.tensorCacheCPU.numpy()

        cdef int idx
        for idx in range(thisBatchSize):
            # first create the data in the cpu memory staging area
            batch[idx].encodeIntoTensor(npTensor, idx, False)
        
        # copy the data to the GPU in one go
        self.forwardInputGPU[:thisBatchSize] = self.tensorCacheCPU[:thisBatchSize]

    def runNetwork(self, int thisBatchSize, asyncCall):
        with torch.no_grad():
            # TODO could probably form an extra net somehow that does not include the replyP head at all?!
            packed = self.net(self.forwardInputGPU[:thisBatchSize])
            if len(packed) > 3:
                moveP, winP, replyP, e1, e2 = packed
            else:
                moveP, winP, replyP = packed
        
        if asyncCall is not None:
            asyncCall()

        return torch.exp(winP), torch.exp(moveP)

    def packageResults(self, list batch, int thisBatchSize, float[:,:] winTensor, object moveTensor):
        cdef list results = []

        cdef int pcount = self.protoState.getPlayerCount()
        cdef int pid, mappedIndex, idx

        allResultsData = np.zeros((thisBatchSize, pcount+1), dtype=np.float32)

        for idx in range(thisBatchSize):
            state = batch[idx]
            movesResult = moveTensor[idx]
            winsResult = allResultsData[idx]
            winsResult[0] = winTensor[idx, 0]
            for pid in range(1, pcount+1):
                mappedIndex = state.mapPlayerNumberToTurnRelative(pid) + 1
                winsResult[pid] = winTensor[idx, mappedIndex]

            results.append((movesResult, winsResult))

        return results

    def innerForward(self, list batch, asyncCall):
        cdef int thisBatchSize = len(batch)
        
        if thisBatchSize == 0:
            if asyncCall is not None:
                asyncCall()
            return []

        self.prepareForwardData(batch, thisBatchSize)

        winP, moveP = self.runNetwork(thisBatchSize, asyncCall)

        moveTensor = moveP.cpu().numpy()
        cdef float[:,:] winTensor = winP.cpu().numpy()

        return self.packageResults(batch, thisBatchSize, winTensor, moveTensor)

    def forward(self, batch, asyncCall = None):
        cdef int nbatch = len(batch)
        cdef int batchNum = math.ceil(nbatch / self.batchSize)
        
        if batchNum == 0 and (asyncCall is not None):
            asyncCall()
            return []

        cdef list results = []
        cdef int bi, batchStart, batchEnd
        cdef list fresult
        for bi in range(batchNum):
            batchStart = bi * self.batchSize
            batchEnd = min((bi+1) * self.batchSize, nbatch)
            fresult = self.innerForward(batch[batchStart:batchEnd], asyncCall)
            # after the first call do not call it again in the next batch!
            asyncCall = None
            for fr in fresult:
                results.append(fr)
        return results

    def getUUID(self):
        return self.uuid

    def reset(self):
        self.uuid = str(uuid.uuid4())
        self.initNetwork()

    def getLr(self):
        result = 0.0001
        for param_group in self.optimizer.param_groups:
            if "lr" in param_group:
                result = param_group["lr"]
                break
        return result

    def setLr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def getMomentum(self):
        result = None
        for param_group in self.optimizer.param_groups:
            if "momentum" in param_group:
                result = param_group["momentum"]
                break
        return result

    def setMomentum(self, momentum):
        if momentum is None:
            return

        for param_group in self.optimizer.param_groups:
            param_group["momentum"] = momentum

    def prepareExample(self, frame):
        return self.packer.prepareExample(frame)

    def packageExamplesBatch(self, examples):
        return self.packer.packageExamplesBatch(examples)

    def getExamplePrepareObject(self):
        return PytorchExamplePrepareWorker(self.device, self.protoState.getGameConstructorName(),\
            self.protoState.getGameConstructorParams(), self.gameDims, self.replyWeight > 0, self.useWinFeatures, self.useMoveFeatures)

    def fit(self, data, iteration = None, iterationProgress = None, forceLr = None):
        self.uuid = str(uuid.uuid4())

        prevLr = self.getLr()
        prevMomentum = self.getMomentum()

        if forceLr is not None:
            self.setLr(forceLr)
        elif iteration is not None and self.lrDecider is not None and iterationProgress is not None:
            lrv = self.lrDecider.getValue(iteration, iterationProgress)
            self.setLr(lrv)

        if iteration is not None and self.momentumDecider is not None and iterationProgress is not None:
            self.setMomentum(self.momentumDecider.getValue(iteration, iterationProgress))

        self.net.train(True)

        cdef int examplesCount, batchNum, bi, batchStart, batchEnd, thisBatchSize

        cdef int sbatchSize = self.batchSize
        cdef float vlw = self.valueLossWeight

        cdef float wOpp = self.replyWeight

        if self.useWinFeatures != -1 or self.useMoveFeatures != -1:
            winFeatures = data[len(data)-2]
            moveFeatures = data[len(data)-1]
            data = data[:len(data)-2]

        if self.replyWeight > 0:
            nIn, mT, wT, rT, examplesCount = data
        else:
            nIn, mT, wT, examplesCount = data

        batchNum = math.ceil(examplesCount / self.batchSize)

        epochStart = time.time()
        mls = []
        wls = []
        rls = []

        wflosses = []
        mflosses = []

        uwf = self.useWinFeatures
        if uwf != -1 and (not isinstance(uwf, list)):
            uwf = list(range(0, self.useWinFeatures + 1))

        umf = self.useMoveFeatures
        if umf != -1 and (not isinstance(umf, list)):
            umf = list(range(0, self.useMoveFeatures + 1))

        for bi in range(batchNum):
            batchStart = bi*sbatchSize
            batchEnd = min((bi+1) * sbatchSize, examplesCount)
            thisBatchSize = batchEnd - batchStart

            x = nIn[batchStart:batchEnd]
            yM = mT[batchStart:batchEnd]
            yW = wT[batchStart:batchEnd]

            if wOpp > 0:
                yR = rT[batchStart:batchEnd]

            wfs = []
            if self.useWinFeatures != -1:
                for wfIdx in uwf:
                    wfs.append(winFeatures[wfIdx][batchStart:batchEnd])

            mfs = []
            if self.useMoveFeatures != -1:
                for mfIdx in umf:
                    mfs.append(moveFeatures[mfIdx][batchStart:batchEnd])

            self.optimizer.zero_grad()

            mO, wO, rO, headWin, headMove = self.net(x)

            mLoss = -torch.sum(mO * yM) / thisBatchSize
            wLoss = -torch.sum(wO * yW) / thisBatchSize

            loss = mLoss + (vlw * wLoss)

            if wOpp > 0:
                rLoss = -torch.sum(rO * yR) / thisBatchSize
                loss += wOpp * rLoss 

            mseLoss = nn.MSELoss()
            for widx, wf in enumerate(wfs):
                wfloss = mseLoss(wf, headWin[:,widx,:, :].view(thisBatchSize, -1))
                wflosses.append(wfloss.data.item())
                loss += self.featuresWeight * wfloss

            for midx, mf in enumerate(mfs):
                # when training on the res block output headMove == headWin, so go from the other end of the shape to prevent a conflict
                matchToIndex = headMove.shape[1]-midx-1
                mfloss = mseLoss(mf, headMove[:,matchToIndex,:,:].view(thisBatchSize, -1))
                mflosses.append(mfloss.data.item())
                loss += self.featuresWeight * mfloss

            loss.backward()

            if self.gradClipValue is not None:
                clip_grad_norm_(self.net.parameters(), self.gradClipValue)

            self.optimizer.step()

            mls.append(mLoss.data.item())
            wls.append(wLoss.data.item())

            if wOpp > 0:
                rls.append(rLoss.data.item())

            del mO
            del wO
            del rO
            del mLoss
            del wLoss
            if wOpp > 0:
                del rLoss
            del loss
        
        if not self.silent:
            logMsg("Completed fit with loss (Moves) %f + (Winner) %f in %f seconds" % (np.mean(mls), np.mean(wls), time.time() - epochStart))

        del nIn
        del mT
        del wT
        if wOpp > 0:
            del rT

        # not sure if this is still needed, this comes from some much older version of the code that worked very differently.
        #torch.cuda.empty_cache()

        self.setLr(prevLr)
        self.setMomentum(prevMomentum)

        self.net.train(False)

        return mls, wls, rls, wflosses, mflosses

    def load(self, packed):
        uuid, stateDict, netConfig = unpackTorchNetwork(packed)

        if len(netConfig) > 0:
            [self.blocks, self.filters, self.headKernel, self.headFilters, self.networkMode] = netConfig
            self.reset()

        self.uuid = uuid

        self.net.load_state_dict(stateDict)
        self.net = self.net.to(self.device)

        self.net.train(False)

    def store(self):
        versionKey = np.array([255], dtype=np.uint8)

        filtersHigh = (self.filters >> 8) & 255
        filtersLow = (self.filters) & 255

        headFiltersHigh = (self.headFilters >> 8) & 255
        headFiltersLow = (self.headFilters) & 255

        netConfig = np.array([self.blocks, filtersHigh, filtersLow, self.headKernel, headFiltersHigh, headFiltersLow, 1 if self.networkMode == "plain" else 0], np.uint8)
        uuidBytes = stringToBytes(self.uuid)
        ublen = np.array([uuidBytes.shape[0]], dtype=np.uint8)

        buffer = io.BytesIO()
        torch.save(self.net.state_dict(), buffer)

        modelBytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

        result = np.concatenate((versionKey, netConfig, ublen, uuidBytes, modelBytes))

        return result

    def upgrade(self):
        """
        upgrading this policy to a bigger network. The upgraded policy will need new training
        """
        self.filters *= 2
        self.reset()

