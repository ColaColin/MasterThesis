# cython: profile=False

import bson
import numpy as np
import gzip

# I don't see much chance to improve this, it's just slow.
# maybe use cpickle after all?! though it seems this can encode ~10k connect4 states per second,
# so compared to generating 10k states it is negligible, as states in connect4 are generated at maybe a 100 a second

def encodeToBson(npDict):
    def pack(d):
        cdef int i

        if isinstance(d, np.ndarray):
            if d.dtype == np.uint8:
                return bytes(d)
            else:
                tmp = dict()
                tmp["_x_encType"] = "numpy"
                tmp["type"] = str(d.dtype)
                tmp["data"] = d.tolist()
                return tmp
        if isinstance(d, dict):
            newDict = dict()
            for k in d:
                newDict[k] = pack(d[k])
            return newDict
        elif isinstance(d, tuple):
            return tuple(map(lambda x: pack(x), d))
        elif isinstance(d, list):
            newList = list()
            for i in range(len(d)):
                newList.append(pack(d[i]))
            return newList

        return d

    npDict = pack(npDict)

    cdef int i

    # the bson lib does not like packing anything but a dict at the root node?!
    if not isinstance(npDict, dict):
        wrapDict = dict()
        wrapDict["_x_encType"] = "dict"
        wrapDict["data"] = npDict
        npDict = wrapDict

    binary = bson.dumps(npDict)
    return gzip.compress(binary)

def decodeFromBson(encBytes):
    def unpack(d):
        cdef int i

        if isinstance(d, dict):
            if "_x_encType" in d:
                if d["_x_encType"] == "numpy":
                    return np.array(d["data"], dtype=d["type"])
                else:
                    assert False, "Unknown enctype: " + d["_x_encType"]
            else:
                for k in d:
                    d[k] = unpack(d[k])
        elif isinstance(d, bytes):
            return np.frombuffer(d, dtype=np.uint8)
        elif isinstance(d, tuple):
            return tuple(map(lambda x: unpack(x), d))
        elif isinstance(d, list):
            for i in range(len(d)):
                d[i] = unpack(d[i])

        return d

    encDict = gzip.decompress(encBytes)
    loaded = bson.loads(encDict)
    
    cdef int i
    if "_x_encType" in loaded and loaded["_x_encType"] == "dict":
        loaded = loaded["data"]

    return unpack(loaded)