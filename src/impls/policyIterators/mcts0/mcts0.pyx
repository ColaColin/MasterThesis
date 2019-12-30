# cython: profile=False

from core.policy.PolicyIterator import PolicyIterator

from utils.fields.fields cimport initField, writeField, printField

import abc


def say_hello_to(name):

    cdef signed char* tmp = initField(7, 6, 0)
    writeField(tmp, 7, 4, 1, 2)

    printField(7, 6, tmp)

    print("Hello %s!!!!" % name)

class MctsPolicyIterator(PolicyIterator, metaclass=abc.ABCMeta):

    def iteratePolicy(self, policy, gamesBatch):
        pass

    def testFunc(self):
        print("test!")