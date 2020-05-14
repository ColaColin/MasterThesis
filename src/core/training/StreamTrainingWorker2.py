import time

from utils.prints import logMsg

from utils.bsonHelp.bsonHelp import decodeFromBson

import abc

import random

import numpy as np


# this better version of the StreamTrainingWorker2 also uses the WindowSizeManager defined in StreamTrainingWorker.py


class StreamTrainingWorker2():
    def __init__(self, policy, windowManager, batchSize):
        self.policy = policy
        self.batchSize = batchSize

    def main(self):
        pass