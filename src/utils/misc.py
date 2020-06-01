import importlib

import os
from pathlib import Path
import json
import abc

def constructor_for_class_name(module_name):
    class_name = module_name.split(".")[-1]
    m = importlib.import_module(".".join(module_name.split(".")[:-1]))
    c = getattr(m, class_name)
    return c

def storeFileUnderPath(path, binaryContent):
    """
    store binary data under path
    """
    dirName = os.path.dirname(path)
    Path(dirName).mkdir(parents=True, exist_ok=True)
    with open(path, "w+b") as f:
        f.write(binaryContent)

def readFileUnderPath(path):
    with open(path, "rb") as f:
        return f.read()

def openJsonFile(path):
    with open(path) as f:
        return json.load(f)

def writeJsonFile(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)

def rPad(sx, padLen):
    result = sx
    while (len(result) < padLen):
        result += " "
    return result

def hConcatStrings(strs):
    blocks = [s.split("\n") for s in strs]
    maxHeight = max(*[len(b) for b in blocks])
    blockWidths = [max(*[len(l) for l in b]) for b in blocks]
    result = ""

    print(maxHeight, blockWidths)

    for h in range(maxHeight):
        for bwidth, block in zip(blockWidths, blocks):
            if len(block) > h:
                result += rPad(block[h], bwidth)
            else:
                result += rPad("", bwidth)
        result += "\n"

    return result


class IterationCalculatedValue(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getValue(self, iteration, iterationProgress):
        """
        @return a value calculated from the current iteration (an integer) and iterationProgress (a float between 0 and 1)
        """