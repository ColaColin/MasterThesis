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
    def __init__(self, inWidth, inHeight, inDepth, baseKernelSize, baseFeatures, features, blocks, moveSize, winSize, extraHeadFilters):
        super(ResCNN, self).__init__()

        paddingSize = 1 if baseKernelSize > 2 else 0

        self.baseConv = nn.Conv2d(inDepth, baseFeatures, baseKernelSize, padding=paddingSize)
        self.baseBn = nn.BatchNorm2d(baseFeatures)
        self.act = nn.ReLU(inplace=True)
        
        if (baseFeatures != features and blocks > 0):
            self.matchConv = nn.Conv2d(baseFeatures, features, 1)
        else:
            self.matchConv = None
        
        blockList = []
        for _ in range(blocks):
            blockList.append(ResBlock(features))
        self.resBlocks = nn.Sequential(*blockList)

        self.extraHeadFilters = extraHeadFilters;
        if self.extraHeadFilters is not None:
            self.moveHeadConv = nn.Conv2d(features, self.extraHeadFilters, 3, padding=1)
            self.valueHeadConv = nn.Conv2d(features, self.extraHeadFilters, 3, padding=1)
            hiddens = self.extraHeadFilters * (inWidth - (baseKernelSize - (1 + paddingSize * 2))) * (inHeight - (baseKernelSize - (1 + paddingSize * 2)))
        else:
            hiddens = features * (inWidth - (baseKernelSize - (1 + paddingSize * 2))) * (inHeight - (baseKernelSize - (1 + paddingSize * 2)))

        self.moveHead = nn.Linear(hiddens, moveSize)
        
        if winSize > 0:
            self.winHead = nn.Linear(hiddens, winSize)
        else:
            self.winHead = None
            
        self.lsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.act(self.baseBn(self.baseConv(x)))
        
        if (self.matchConv != None):
            x = self.matchConv(x)
        
        x = self.resBlocks(x)
        
        if self.extraHeadFilters is None:
            x = x.view(x.size(0), -1)

            moveP = self.lsoftmax(self.moveHead(x))

            if self.winHead != None:
                winP = self.lsoftmax(self.winHead(x))
                return moveP, winP
            else:
                return moveP
        else:
            moveX = self.moveHeadConv(x)
            winX = self.valueHeadConv(x)

            moveX = moveX.view(moveX.size(0), -1)
            winX = winX.view(winX.size(0), -1)

            moveP = self.lsoftmax(self.moveHead(moveX))

            if self.winHead != None:
                winP = self.lsoftmax(self.winHead(winX))
                return moveP, winP
            else:
                return moveP
        
def gameResultsToAbsoluteWinTensor(wins, playerCount):
    result = np.zeros(playerCount, dtype=np.float32)
    for win in wins:
        result[win] += 1
    result /= np.sum(result)
    return result

class PytorchExamplePrepareWorker(ExamplePrepareWorker, metaclass=abc.ABCMeta):
    def __init__(self, device, gameCtor, gameParams, gameDims):
        self.device = device
        self.gameCtor = gameCtor
        self.gameParams = gameParams
        self.protoState = None
        self.gameDims = gameDims

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

        result = [inputArray, outputMoves, outputWins, hash(game)]
        return result

    def getHashForExample(self, example):
        return example[3]

    def areExamplesEqual(self, exampleA, exampleB):
        return torch.equal(exampleA[0], exampleB[0])

    def mergeInto(self, target, source, sourceWeight):
        targetWeight = 1 - sourceWeight
        target[1] *= targetWeight
        target[2] *= targetWeight
        target[1] += sourceWeight * source[1]
        target[2] += sourceWeight * source[2]

    def packageExamplesBatch(self, examples):
        # format is (inputs, movesOut, winsOut, examplesCount)[]
        random.shuffle(examples)
        inputs = torch.cat(list(map(lambda x: x[0], examples))).to(self.device)
        movesOut = torch.cat(list(map(lambda x: x[1], examples))).to(self.device)
        winsOut = torch.cat(list(map(lambda x: x[2], examples))).to(self.device)
        return (inputs, movesOut, winsOut, len(examples))
    

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
    def __init__(self, peak, end, baseVal, peakVal, endVal, dbgName="one_cycle"):
        self.peak = peak
        self.end = end
        self.baseVal = baseVal
        self.peakVal = peakVal
        self.endVal = endVal
        self.lastPhase = "end" # inc -> dec -> end
        self.name = dbgName

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
            return result
        elif currentPhase == "dec":
            progress = (iterationProgress - self.peak) / (self.end - self.peak)
            result = self.peakVal + progress * (self.baseVal - self.peakVal)
            return result
        elif currentPhase == "end":
            progress = (iterationProgress - self.end) / (1 - self.end)
            result = self.baseVal + progress * (self.endVal - self.baseVal)
            return result

        assert False, ("unknown currentPhase value: " + currentPhase)

class PytorchPolicy(Policy, metaclass=abc.ABCMeta):
    """
    A policy that uses Pytorch to implement a ResNet-tower similar to the one used by the original AlphaZero implementation.
    """

    def __init__(self, batchSize, blocks, filters, headKernel, headFilters, protoState, device, optimizerName, \
            optimizerArgs = None, extraHeadFilters = None, silent = True, lrDecider = None, gradClipValue = None, valueLossWeight = 1, momentumDecider = None):
        self.batchSize = batchSize
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

        self.initNetwork()

        self.lrDecider = lrDecider
        self.momentumDecider = momentumDecider

        self.packer = PytorchExamplePrepareWorker(self.device, self.protoState.getGameConstructorName(), self.protoState.getGameConstructorParams(), self.gameDims)

        logMsg("\nCreated a PytorchPolicy using device " + str(self.device) + "\n", pms.summary(self.net, torch.zeros((1, ) + self.gameDims, device = self.device)))

    def initNetwork(self):
        self.net = ResCNN(self.gameDims[1], self.gameDims[2], self.gameDims[0],\
            self.headKernel, self.headFilters, self.filters, self.blocks,\
            self.protoState.getMoveCount(), self.protoState.getPlayerCount() + 1, self.extraHeadFilters)
        self.net = self.net.to(self.device)

        # can't use mlconfig, as mlconfig has no access to self.net.parameters :(
        octor = constructor_for_class_name(self.optimizerName)
        self.optimizer = octor(self.net.parameters(), **self.optimizerArgs)
        self.net.train(False)

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
            moveP, winP = self.net(self.forwardInputGPU[:thisBatchSize])
        
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
        return PytorchExamplePrepareWorker(self.device, self.protoState.getGameConstructorName(), self.protoState.getGameConstructorParams(), self.gameDims)

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

        nIn, mT, wT, examplesCount = data

        batchNum = math.ceil(examplesCount / self.batchSize)

        epochStart = time.time()
        mls = []
        wls = []
        
        for bi in range(batchNum):
            batchStart = bi*sbatchSize
            batchEnd = min((bi+1) * sbatchSize, examplesCount)
            thisBatchSize = batchEnd - batchStart

            x = nIn[batchStart:batchEnd]
            yM = mT[batchStart:batchEnd]
            yW = wT[batchStart:batchEnd]

            self.optimizer.zero_grad()

            mO, wO = self.net(x)

            mLoss = -torch.sum(mO * yM) / thisBatchSize
            wLoss = -torch.sum(wO * yW) / thisBatchSize

            loss = mLoss + (vlw * wLoss)
            loss.backward()

            if self.gradClipValue is not None:
                clip_grad_norm_(self.net.parameters(), self.gradClipValue)

            self.optimizer.step()

            mls.append(mLoss.data.item())
            wls.append(wLoss.data.item())
            del mO
            del wO
            del mLoss
            del wLoss
            del loss
        
        if not self.silent:
            logMsg("Completed fit with loss (Moves) %f + (Winner) %f in %f seconds" % (np.mean(mls), np.mean(wls), time.time() - epochStart))

        del nIn
        del mT
        del wT

        # not sure if this is still needed, this comes from some much older version of the code that worked very differently.
        #torch.cuda.empty_cache()

        self.setLr(prevLr)
        self.setMomentum(prevMomentum)

        self.net.train(False)

        return mls, wls

    def load(self, packed):
        ublen = packed[0]
        uuidBytes = packed[1:ublen+1]
        modelBuffer = io.BytesIO(bytes(packed[ublen+1:]))

        self.uuid = bytesToString(uuidBytes)

        self.net.load_state_dict(torch.load(modelBuffer, map_location=torch.device('cpu')))
        self.net = self.net.to(self.device)

        self.net.train(False)

    def store(self):
        uuidBytes = stringToBytes(self.uuid)
        ublen = np.array([uuidBytes.shape[0]], dtype=np.uint8)

        buffer = io.BytesIO()
        torch.save(self.net.state_dict(), buffer)

        modelBytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

        result = np.concatenate((ublen, uuidBytes, modelBytes))

        return result


