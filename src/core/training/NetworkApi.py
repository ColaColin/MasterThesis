import time
import sys

from utils.bsonHelp.bsonHelp import decodeFromBson, encodeToBson
from utils.prints import logMsg

import requests

class NetworkApi():
    def __init__(self):
        hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for the trainer: --secret <server password>, --run <uuid> and --command <command server host>!")

        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.runId = sys.argv[sys.argv.index("--run")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]

    def loadNewestNetwork(self, policy):
        networkList = self.getNetworkList()
        if len(networkList) > 0:
            networkList.sort(key=lambda x: x["creation"], reversed=True)
            logMsg("Continue training of an existing network", networkList[0])
            networkId = networkList[0]["id"]

            networkData = self.downloadNetwork(networkId)

            policy.load(networkData)

            logMsg("Network %s loaded" % policy.getUUID())
        

    def getNetworkList(self):
        while True:
            try:
                networkList = requests.get(url=self.commandHost + "/api/networks/list/" + self.runId, headers={"secret": self.secret}).json()
                return networkList
            except Exception as error:
                logMsg("Could not get network list, hoping this is a temporary failure, will try again soon!", error)
                time.sleep(10)

    def uploadNetwork(self, policy):
        while True:
            try:
                newPolicyEncoded = encodeToBson(policy.store())
                response = requests.post(url=self.commandHost + "/api/networks/" + self.runId + "/" + policy.getUUID(), data=newPolicyEncoded,
                    headers={"secret": self.secret})
                response.raise_for_status()
                logMsg("Network uploaded successfully!")
                break
            except Exception as error:
                logMsg("Could not upload network, hoping this is a temporary failure, will try again soon!", error)
                time.sleep(10)
    
    def downloadNetwork(self, networkId):
        while True:
            try:
                response = requests.get(url=self.commandHost + "/api/networks/download/" + networkId, stream=True, headers={"secret": self.secret})
                response.raise_for_status()
                networkData = decodeFromBson(response.raw.data)
                return networkData
            except Exception as error:
                logMsg("Could not download network, will try again soon", error)
                time.sleep(10)

