# cython: profile=False

import bson
import numpy as np
import gzip

# I don't see much chance to improve this, it's just slow.

def encodeToBson(npDict):
    def pack(d):
        cdef int i

        if isinstance(d, dict):
            for k in d:
                if isinstance(d[k], np.ndarray):
                    if d[k].dtype == np.uint8:
                        d[k] = bytes(d[k])
                    else:
                        tmp = dict()
                        tmp["___encType"] = "numpy"
                        tmp["type"] = str(d[k].dtype)
                        tmp["data"] = d[k].tolist()
                        d[k] = tmp
                elif isinstance(d[k], dict) or isinstance(d[k], list):
                    d[k] = pack(d[k])
        elif isinstance(d, list):
            for i in range(len(d)):
                d[i] = pack(d[i])
        return d

    npDict = pack(npDict)

    cdef int i

    if isinstance(npDict, list):
        wrapDict = dict()
        wrapDict["___encType"] = "list"
        for i in range(len(npDict)):
            wrapDict[i] = npDict[i]
        
        npDict = wrapDict

    binary = bson.dumps(npDict)
    return gzip.compress(binary)

def decodeFromBson(encBytes):
    def unpack(d):
        cdef int i

        if isinstance(d, dict):
            for k in d:
                if isinstance(d[k], dict):
                    if "___encType" in d[k]:
                        if d[k]["___encType"] == "numpy":
                            d[k] = np.array(d[k]["data"], dtype=d[k]["type"])
                        else:
                            assert False, "Unknown enctype: " + d[k]["___encType"]
                    else:
                        d[k] = unpack(d[k])
                elif isinstance(d[k], dict) or isinstance(d[k], list):
                    d[k] = unpack(d[k])
                elif isinstance(d[k], bytes):
                    d[k] = np.frombuffer(d[k], dtype=np.uint8)
        elif isinstance(d, list):
            for i in range(len(d)):
                d[i] = unpack(d[i])

        return d

    encDict = gzip.decompress(encBytes)
    loaded = bson.loads(encDict)
    
    cdef int i
    if "___encType" in loaded and loaded["___encType"] == "list":
        tmp = [None] * (len(loaded) - 1)
        for k in range(len(tmp)):
            tmp[k] = loaded[str(k)]
        loaded = tmp

    return unpack(loaded)