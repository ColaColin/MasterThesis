import numpy as np

def stringToBytes(txt):
    return np.array([ord(t) for t in txt], dtype=np.uint8)

def bytesToString(bytes):
    return ''.join([chr(b) for b in bytes])