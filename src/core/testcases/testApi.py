import unittest
import subprocess

import sys
import os
import shutil
import json
import time
import psycopg2
from psycopg2 import pool

import requests

from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson
from impls.games.connect4.connect4 import Connect4GameState
import numpy as np

import uuid

from impls.polices.pytorch.policy import PytorchPolicy

from utils.prints import logMsg, setLoggingEnabled

# This test requires a database called x0_test with user x0_test and password x0_test to exist in a local postgres installation
# That database will be filled with tables and cleared

config = {
    "configFile": "/tmp/x0_test_config.json",
    "dataPath": "/tmp/x0_test_data_directory/",
    "secret": "42!",
    "dbuser": "x0_test",
    "dbpassword": "x0_test",
    "dbname": "x0_test",
    "host": "127.0.0.1",
    "port": 7877,
    "nostats": True
}

urlBase = "http://" + config["host"] + ":" + str(config["port"])+ "/"

def makeReport():
    game = Connect4GameState(7,6,4)
    game = game.playMove(np.random.randint(7))
    game = game.playMove(np.random.randint(7))

    report = dict()
    report["knownResults"] = [1]
    report["generics"] = dict()
    report["policyUUID"] = "ab98246f-4b80-48e8-97fc-d365d4a3aa3d"
    report["policyIterated"] = np.random.rand(7).astype(np.float32)
    report["uuid"] = str(uuid.uuid4())
    report["state"] = game.store()

    return report

class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prntVerbose = ('-v' in sys.argv) or ('--verbose' in sys.argv)
        setLoggingEnabled(prntVerbose)

        logMsg("Working with tests data in " + config["dataPath"])
        with open(config["configFile"], "w") as f:
            json.dump(config, f)
        
        silence = subprocess.DEVNULL
        if prntVerbose:
            silence = None

        cls.process = subprocess.Popen(["python", "-m", "core.mains.command", "--config", config["configFile"]],
            stdout=silence, stderr=silence)
        # wait a moment so the process is started, 1 second should be plenty
        time.sleep(1.5)
    
    @classmethod
    def tearDownClass(cls):
        try:
            cls.process.kill()
        except Exception as error:
            logMsg("Could not kill api process", error)

        try:
            os.remove(config["configFile"])
        except Exception as error:
            logMsg("could not delete test config!", error)
        
        try:
            shutil.rmtree(config["dataPath"])
        except Exception as error:
            logMsg("could not delete tmp test directory!", error)

        logMsg("Cleanup for ApiTest completed")

    def setUp(self):
        resetSql = os.path.join(os.getcwd(), "setup.sql")

        self.pool = psycopg2.pool.SimpleConnectionPool(1, 3,user = config["dbuser"],
                                              password = config["dbpassword"],
                                              host = "127.0.0.1",
                                              port = "5432",
                                              database = config["dbname"]);

        with open(resetSql, "r") as f:
            sqlString = f.read()

        try:
            con = self.pool.getconn()
            cursor = con.cursor()
            cursor.execute(sqlString)
            con.commit()
        finally:
            if cursor:
                cursor.close()
            self.pool.putconn(con)

    def tearDown(self):
        self.pool.closeall()
        
    def test_missing_password(self):
        response = requests.get(url=urlBase + "api/runs")
        # no password was provided -> access should be denied
        self.assertEqual(response.status_code, 401)        

    def test_password(self):
        response = requests.get(url=urlBase + "api/runs", headers={"secret": config["secret"]})
        self.assertEqual(response.status_code, 200)
        
    def postARun(self):
        rundata = {
            "name": "testRun",
            "config": "This is a Config\nIt has multiple lines of text"
        }
        runjs = json.dumps(rundata)

        response = requests.post(url=urlBase + "api/runs/", data=runjs, headers= {
            "secret": config["secret"],
            "Content-Type": "application/json;charset=utf-8"
        })
        self.assertEqual(response.status_code, 200)
        rundata["id"] = response.json()
        return rundata

    def test_create_run(self):
        rundata = self.postARun()

        response = requests.get(url=urlBase + "api/runs", headers={"secret": config["secret"]})
        self.assertEqual(response.status_code, 200)

        runsList = response.json()

        self.assertEqual(len(runsList), 1)
        self.assertEqual(runsList[0]["id"], rundata["id"])
        self.assertEqual(runsList[0]["name"], rundata["name"])
        self.assertEqual(runsList[0]["config"], rundata["config"])

        response = requests.get(url=urlBase + "/api/runs/" + rundata["id"], headers={"secret": config["secret"]})
        response.raise_for_status()

        post = response.json()
        self.assertEqual(rundata["id"], post["id"])
        self.assertEqual(rundata["name"], post["name"])
        self.assertEqual(rundata["config"], post["config"])

    def test_state_posting(self):
        run1 = self.postARun()
        run2 = self.postARun()

        states1 = [makeReport() for _ in range(123)]
        states2 = [makeReport() for _ in range(456)]

        report1Id = requests.post(url=urlBase + "api/state/test/" + run1["id"],
            data=encodeToBson(states1), headers={"secret": config["secret"]}).json()

        report2Id = requests.post(url=urlBase + "api/state/test2/" + run2["id"],
            data=encodeToBson(states2), headers={"secret": config["secret"]}).json()

        listRun1 = requests.get(url=urlBase + "api/state/list/" + run1["id"], headers={"secret": config["secret"]}).json()

        listRun2 = requests.get(url=urlBase + "api/state/list/" + run2["id"], headers={"secret": config["secret"]}).json()

        self.assertEqual(len(listRun1), 1)
        self.assertEqual(len(listRun2), 1)

        self.assertEqual(listRun1[0]["id"], report1Id)
        self.assertEqual(listRun2[0]["id"], report2Id)

        self.assertEqual(listRun1[0]["worker"], "test")
        self.assertEqual(listRun2[0]["worker"], "test2")

        self.assertEqual(listRun1[0]["packageSize"], len(states1))
        self.assertEqual(listRun2[0]["packageSize"], len(states2))

        response = requests.get(url=urlBase + "api/state/download/" + report1Id, stream=True, headers={"secret": config["secret"]})
        response.raise_for_status()

        states1Downloaded = decodeFromBson(response.raw.data)

        response = requests.get(url=urlBase + "api/state/download/" + report2Id, stream=True, headers={"secret": config["secret"]})
        response.raise_for_status()

        states2Downloaded = decodeFromBson(response.raw.data)

        self.assertEqual(len(states1), len(states1Downloaded))
        self.assertEqual(len(states2), len(states2Downloaded))

        def checkStatesEqual(statesPre, statesPost):
            for pre, post in zip(statesPre, statesPost):
                piPre = pre["policyIterated"]
                piPost = post["policyIterated"]
                stPre = pre["state"]
                stPost = post["state"]

                del pre["policyIterated"]
                del post["policyIterated"]
                del pre["state"]
                del post["state"]

                self.assertDictEqual(pre, post)

                self.assertTrue(np.all(piPre == piPost))
                self.assertTrue(np.all(stPre == stPost))

        checkStatesEqual(states1, states1Downloaded)
        checkStatesEqual(states2, states2Downloaded)

    def test_network_posting(self):
        run = self.postARun()

        policy = PytorchPolicy(32, 1, 32, 3, 64, Connect4GameState(7,6,4), "cuda:0", "torch.optim.adamw.AdamW", {
            "lr": 0.001,
            "weight_decay": 0.0001
        })

        response = requests.post(url=urlBase + "api/networks/" + run["id"] + "/" + policy.getUUID(), data=encodeToBson(policy.store()),
            headers={"secret": config["secret"]})
        response.raise_for_status()

        networkList = requests.get(url=urlBase + "api/networks/list/" + run["id"], headers={"secret": config["secret"]}).json()
        self.assertEqual(len(networkList), 1)
        self.assertEqual(networkList[0]["id"], policy.getUUID())

        response = requests.get(url=urlBase + "api/networks/download/" + policy.getUUID(), stream=True, headers={"secret": config["secret"]})
        response.raise_for_status()

        redownloaded = decodeFromBson(response.raw.data)

        game = Connect4GameState(7,6,4)
        game = game.playMove(np.random.randint(7))
        game = game.playMove(np.random.randint(7))

        policy.isRandom = False
        forwardResultPre = policy.forward([game])[0]
        preUUID = policy.getUUID()
        policy.reset()
        policy.isRandom = False
        forwardResultReset = policy.forward([game])[0]
        policy.load(redownloaded)
        policy.isRandom = False
        forwardResultPost = policy.forward([game])[0]

        self.assertEqual(preUUID, policy.getUUID())

        self.assertTrue(np.all(forwardResultPre[0] == forwardResultPost[0]))
        self.assertTrue(np.all(forwardResultPre[1] == forwardResultPost[1]))

        self.assertFalse(np.all(forwardResultPre[0] == forwardResultReset[0]))
        self.assertFalse(np.all(forwardResultPre[1] == forwardResultReset[1]))