import threading
import requests

import sys
import os
import time

from utils.prints import logMsg

from utils.misc import storeFileUnderPath, readFileUnderPath

from utils.bsonHelp.bsonHelp import decodeFromBson

import json

def mkFoo(self):
    def foo():
        self.run()
    return foo

class StatesDownloader():
    def __init__(self, storageDirectory):
        """
        Requires some configuration parameters to be present in the arguments to python
        --command <command server host>
        --secret <server api password>
        --run <run-uuid>

        Downloads all states belonging to the given run from the given server.
        To be used by the TrainingWorker process to download states in the background. Uses a thread to achieve that.
        So call .start() on this object to make it start working
        """
        self.storageDirectory = storageDirectory
        # id -> object
        self.downloadedStatesObject = {}
        # objects descending sorted by creation time
        self.downloadedStatesHistory = []
        # copy for thread-safe access from the outside
        self.history = []
        self.numStatesAvailable = 0
        self.running = True

        hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for the trainer: --secret <server password>, --run <uuid> and --command <command server host>!")

        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.runId = sys.argv[sys.argv.index("--run")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]

        self.load()

    def store(self):
        with open(os.path.join(self.storageDirectory, "history.json"), "w") as f:
            json.dump(self.downloadedStatesHistory, f)
        
    def load(self):
        path = os.path.join(self.storageDirectory, "history.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.downloadedStatesHistory = json.load(f)
                self.history = self.downloadedStatesHistory.copy()
                logMsg("Found %i state files in the local storage!" % len(self.downloadedStatesHistory))
                self.downloadedStatesObject = {}
                for history in self.downloadedStatesHistory:
                    self.downloadedStatesObject[history["id"]] = history
                    self.numStatesAvailable += history["packageSize"]
            logMsg("Loaded existing local data with %i states in %i packages" % (self.numStatesAvailable, len(self.downloadedStatesHistory)))
        else:
            logMsg("Could not find history.json, assuming this is the first start of the trainer in this location!")

    def start(self):
        t = threading.Thread(target=mkFoo(self))
        t.start()

    def openPackage(self, idName):
        data = readFileUnderPath(os.path.join(self.storageDirectory, idName))
        return decodeFromBson(data)

    def run(self):
        logMsg("Starting states downloader, storing files in", self.storageDirectory)
        while self.running:
            try:
                # first download the current file describing the states on the server
                lresponse = requests.get(url=self.commandHost + "/api/state/list/" + self.runId, headers={"secret": self.secret})
                lresponse.raise_for_status()
                list = lresponse.json()

                sumNewStates = 0
                newEntries = []
                for remoteEntry in list:
                    if not (remoteEntry["id"] in self.downloadedStatesObject):
                        newEntries.append(remoteEntry)
                        sumNewStates += remoteEntry["packageSize"]
                
                # download newest ones first, they are the most interesting
                newEntries.sort(key = lambda x: x["creation"], reverse=True)

                if len(newEntries) > 0:
                    #logMsg("Found %i new state packages with %i states on the server!" % (len(newEntries), sumNewStates))

                    for newEntry in newEntries:
                        response = requests.get(url=self.commandHost + "/api/state/download/" + newEntry["id"], stream=True, headers={"secret": self.secret})
                        response.raise_for_status()
                        statesData = response.raw.data
                        storeFileUnderPath(os.path.join(self.storageDirectory, newEntry["id"]), statesData)
                        self.downloadedStatesObject[newEntry["id"]] = newEntry
                        self.downloadedStatesHistory.append(newEntry)
                        self.downloadedStatesHistory.sort(key = lambda x: x["creation"], reverse=True)
                        self.history = self.downloadedStatesHistory.copy()
                        self.numStatesAvailable += newEntry["packageSize"]
                        self.store()

            except Exception as error:
                logMsg("Could not download states, will try again soon", error)
                time.sleep(10)
            
            time.sleep(5)
