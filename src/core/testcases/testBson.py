import unittest

from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

import numpy as np

class BsonTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_SimpleList(self):
        lst = [1,2,3]
        enc = encodeToBson(lst)
        dec = decodeFromBson(enc)
        self.assertEqual(lst, dec)

    def test_numpy(self):
        data = np.array([1.23456789, 9.87654321])
        transfered = decodeFromBson(encodeToBson(data))
        self.assertEqual(len(data), len(transfered))
        self.assertEqual(np.sum(data == transfered), len(data))

    def test_number(self):
        number = 42
        enc = encodeToBson(number)
        dec = decodeFromBson(enc)
        self.assertEqual(number, dec)

    def test_numpyList(self):
        lst = [np.array([1,2,3]), np.array([4,5,6])]
        transfered = decodeFromBson(encodeToBson(lst))
        self.assertEqual(len(transfered), 2)
        self.assertEqual(transfered[0].tolist(), [1,2,3])
        self.assertEqual(transfered[1].tolist(), [4,5,6])

    def test_dicts(self):
        root = dict()
        root["A"] = 1
        root["B"] = [1,2,3]

        transfered = decodeFromBson(encodeToBson(root))

        self.assertEqual(transfered, root)

    def test_deepNumpyDicts(self):
        root = dict()
        root["A"] = 1
        root["B"] = dict()
        root["B"]["FOO"] = [1,2,3]
        root["C"] = dict()
        root["C"]["data"] = [np.array([3.1415, -3.1415], dtype=np.float32), 42]
        root["D"] = [dict(), dict()]
        root["D"][0]["A"] = 42
        root["D"][1]["B"] = "hello world"

        enc = encodeToBson(root)
        transfered = decodeFromBson(enc)
        self.assertEqual(transfered["A"], root["A"])
        self.assertEqual(transfered["B"], root["B"])
        self.assertTrue("C" in transfered and "data" in transfered["C"] and isinstance(transfered["C"]["data"], list))
        self.assertTrue(np.sum(transfered["C"]["data"][0] == root["C"]["data"][0]) == 2)
        self.assertEqual(transfered["C"]["data"][1], root["C"]["data"][1])
        self.assertTrue(transfered["D"], root["D"])