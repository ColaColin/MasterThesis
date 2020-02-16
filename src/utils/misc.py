import importlib

import os
from pathlib import Path

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