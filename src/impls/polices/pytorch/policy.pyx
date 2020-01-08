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

from core.policy.Policy import Policy

from utils.encoding import stringToBytes, bytesToString

import pytorch_model_summary as pms

import io

import random

import math

import time

from utils.misc import constructor_for_class_name

from utils.prints import logMsg


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
    def __init__(self, inWidth, inHeight, inDepth, baseKernelSize, baseFeatures, features, blocks, moveSize, winSize):
        super(ResCNN, self).__init__()
        assert (inWidth % 2) == (inHeight % 2), "One would need to check how this network behaves in this situation before using it"
        
        self.baseConv = nn.Conv2d(inDepth, baseFeatures, baseKernelSize)
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

        hiddens = features * (inWidth - (baseKernelSize - 1)) * (inHeight - (baseKernelSize - 1))
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
        
        x = x.view(x.size(0), -1)

        moveP = self.lsoftmax(self.moveHead(x))
        
        if self.winHead != None:
            winP = self.lsoftmax(self.winHead(x))
            return moveP, winP
        else:
            return moveP
        
def gameResultsToAbsoluteWinTensor(wins, playerCount):
    result = np.zeros(playerCount, dtype=np.float32)
    for win in wins:
        result[win] += 1
    result /= np.sum(result)
    return result

def fillTrainingSet(protoState, data, int startIndex, moveOutput, winOutput, networkInput):
    if len(data) == 0:
        return
    moveOutput[startIndex : startIndex + len(data)].fill_(0)
    winOutput[startIndex : startIndex + len(data)].fill_(0)
    networkInput[startIndex : startIndex + len(data)].fill_(0)

    cdef float [:,:] moveTensor = moveOutput.numpy()
    cdef float [:,:] winTensor = winOutput.numpy()

    cdef float [:, :, :, :] inputTensor = networkInput.numpy()

    cdef int fidx, idx, mappedIndex, pid

    for fidx, frame  in enumerate(data):
        game = protoState.load(data[fidx]["state"])

        newPolicy = data[fidx]["policyIterated"]

        game.encodeIntoTensor(inputTensor, startIndex + fidx, False)

        for idx, p in enumerate(newPolicy):
            moveTensor[startIndex + fidx, idx] = p
        
        absoluteWinners = gameResultsToAbsoluteWinTensor(data[fidx]["knownResults"], game.getPlayerCount()+1)

        winTensor[startIndex + fidx, 0] = absoluteWinners[0]
        for pid in range(1, game.getPlayerCount()+1):
            mappedIndex = game.mapPlayerNumberToTurnRelative(pid) + 1
            winTensor[startIndex + fidx, mappedIndex] = absoluteWinners[pid]



class PytorchPolicy(Policy, metaclass=abc.ABCMeta):
    """
    A policy that uses Pytorch to implement a ResNet-tower similar to the one used by the original AlphaZero implementation.
    """

    def __init__(self, batchSize, blocks, filters, headKernel, headFilters, protoState, device, optimizerName, optimizerArgs):
        self.batchSize = batchSize
        self.device = torch.device(device)
        
        self.uuid = str(uuid.uuid4())
        self.gameDims = protoState.getDataShape()
        self.protoState = protoState

        self.optimizerName = optimizerName
        self.optimizerArgs = optimizerArgs

        self.blocks = blocks
        self.filters = filters
        self.headKernel = headKernel
        self.headFilters = headFilters
        assert len(self.gameDims) == 3, "ResNet requires a 3D data shape!"
        self.tensorCacheExists = False

        self.initNetwork()

        logMsg("\nCreated a PytorchPolicy using device " + str(self.device) + "\n", pms.summary(self.net, torch.zeros((1, ) + self.gameDims, device = self.device)))

    def initNetwork(self):
        self.net = ResCNN(self.gameDims[1], self.gameDims[2], self.gameDims[0],\
            self.headKernel, self.headFilters, self.filters, self.blocks,\
            self.protoState.getMoveCount(), self.protoState.getPlayerCount() + 1)
        self.net = self.net.to(self.device)

        # can't use mlconfig, as mlconfig has no access to self.net.parameters :(
        octor = constructor_for_class_name(self.optimizerName)
        self.optimizer = octor(self.net.parameters(), **self.optimizerArgs)

    def innerForward(self, batch, asyncCall):
        cdef int thisBatchSize = len(batch)
        
        if thisBatchSize == 0:
            if asyncCall is not None:
                asyncCall()
            return []

        assert self.batchSize >= len(batch), "PytorchPolicy has a max batchSize below " + str(len(batch))

        if not self.tensorCacheExists:
            self.tensorCacheCPU = torch.zeros((self.batchSize, ) + self.gameDims).pin_memory()
            self.forwardInputGPU = Variable(torch.zeros((self.batchSize, ) + self.gameDims), requires_grad=False).to(self.device)
            self.tensorCacheExists = True

        cdef float [:, :, :, :] npTensor = self.tensorCacheCPU.numpy()

        cdef int idx
        for idx in range(thisBatchSize):
            # first create the data in the cpu memory staging area
            batch[idx].encodeIntoTensor(npTensor, idx, False)
        
        # copy the data to the GPU in one go
        self.forwardInputGPU[:thisBatchSize] = self.tensorCacheCPU[:thisBatchSize]

        with torch.no_grad():
            moveP, winP = self.net(self.forwardInputGPU[:thisBatchSize])
        
        if asyncCall is not None:
            asyncCall()
        
        winP = torch.exp(winP)
        moveP = torch.exp(moveP)

        cdef float [:,:] moveTensor = moveP.cpu().numpy()
        cdef float [:,:] winTensor = winP.cpu().numpy()

        results = []

        cdef int pcount = self.protoState.getPlayerCount()
        cdef int pid, mappedIndex

        for idx in range(thisBatchSize):
            state = batch[idx]
            movesResult = np.asarray(moveTensor[idx], dtype=np.float32)
            winsResult = np.zeros(pcount+1, dtype=np.float32)
            winsResult[0] = winTensor[idx, 0]
            for pid in range(1, pcount+1):
                mappedIndex = state.mapPlayerNumberToTurnRelative(pid) + 1
                winsResult[pid] = winTensor[idx, mappedIndex]

            results.append((movesResult, winsResult))

        return results

    def forward(self, batch, asyncCall = None):
        nbatch = len(batch)
        batchNum = math.ceil(nbatch / self.batchSize)
        
        results = []
        for bi in range(batchNum):
            batchStart = bi * self.batchSize
            batchEnd = min((bi+1) * self.batchSize, nbatch)
            fresult = self.innerForward(batch[batchStart:batchEnd], asyncCall)
            for fr in fresult:
                results.append(fr)
        return results
        

    def getUUID(self):
        return self.uuid

    def reset(self):
        self.uuid = str(uuid.uuid4())
        self.initNetwork()

    def fit(self, data, epochs):
        self.uuid = str(uuid.uuid4())

        data = data.copy()
        recordsCount = len(data)
        self.trainingInputs = torch.zeros((recordsCount, ) + self.gameDims)
        self.trainingOutputsMoves = torch.zeros((recordsCount, self.protoState.getMoveCount()))
        self.trainingOutputsWins = torch.zeros((recordsCount, (self.protoState.getPlayerCount() + 1)))
        self.net.train(True)

        logMsg("Preparing epoch of data")
        pStart = time.time()
        random.shuffle(data)
        fillTrainingSet(self.protoState, data, 0, self.trainingOutputsMoves, self.trainingOutputsWins, self.trainingInputs)

        pEnd = time.time()
        logMsg("Preparing %i data samples took %f seconds" % (len(data), time.time() - pStart))

        batchNum = math.ceil(len(data) / self.batchSize)

        for e in range(epochs):
            logMsg("Starting epoch " + str(e + 1))

            epochStart = time.time()
            nIn = Variable(self.trainingInputs.clone().detach()).to(self.device)
            mT = Variable(self.trainingOutputsMoves.clone().detach()).to(self.device)
            wT = Variable(self.trainingOutputsWins.clone().detach()).to(self.device)

            mls = []
            wls = []

            random.shuffle(data)

            for bi in range(batchNum):
                batchStart = bi*self.batchSize
                batchEnd = min((bi+1) * self.batchSize, recordsCount)
                batchSize = batchEnd - batchStart
                batchMiddle = batchStart + (batchSize // 2)

                x = nIn[batchStart:batchEnd]
                yM = mT[batchStart:batchEnd]
                yW = wT[batchStart:batchEnd]

                self.optimizer.zero_grad()

                mO, wO = self.net(x)
                # cpu work performed behind calls to pytorch like this can be hidden behind the gpu work
                fillTrainingSet(self.protoState, data[batchStart:batchMiddle], batchStart, self.trainingOutputsMoves, self.trainingOutputsWins, self.trainingInputs)

                mLoss = -torch.sum(mO * yM) / batchSize
                wLoss = -torch.sum(wO * yW) / batchSize

                loss = mLoss + wLoss
                loss.backward()

                # cpu work performed behind calls to pytorch like this can be hidden behind the gpu work
                fillTrainingSet(self.protoState, data[batchMiddle:batchEnd], batchMiddle, self.trainingOutputsMoves, self.trainingOutputsWins, self.trainingInputs)

                self.optimizer.step()

                mls.append(mLoss.data.item())
                wls.append(wLoss.data.item())
            
            logMsg("Completed Epoch %i with loss (Moves) %f + (Winner) %f in %f seconds" % (e+1, np.mean(mls), np.mean(wls), time.time() - epochStart))
       
        self.net.train(False)

        del self.trainingInputs
        del self.trainingOutputsMoves
        del self.trainingOutputsWins
        torch.cuda.empty_cache()
       

    def load(self, packed):
        ublen = packed[0]
        uuidBytes = packed[1:ublen+1]
        modelBuffer = io.BytesIO(bytes(packed[ublen+1:]))

        self.uuid = bytesToString(uuidBytes)
        self.net.load_state_dict(torch.load(modelBuffer))

    def store(self):
        uuidBytes = stringToBytes(self.uuid)
        ublen = np.array([uuidBytes.shape[0]], dtype=np.uint8)

        buffer = io.BytesIO()
        torch.save(self.net.state_dict(), buffer)

        modelBytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

        result = np.concatenate((ublen, uuidBytes, modelBytes))

        return result


