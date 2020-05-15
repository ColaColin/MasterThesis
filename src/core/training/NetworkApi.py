import time
import sys

from utils.bsonHelp.bsonHelp import decodeFromBson, encodeToBson
from utils.prints import logMsg
from utils.req import requestJson, postBytes, requestBytes

class NetworkApi():
    def __init__(self, noRun=False):
        hasArgs = ("--secret" in sys.argv) and (("--run" in sys.argv) or noRun) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for the trainer: --secret <server password>, --run <uuid> and --command <command server host>!")

        self.secret = sys.argv[sys.argv.index("--secret")+1]
        if not noRun:
            self.runId = sys.argv[sys.argv.index("--run")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]

    def loadNewestNetwork(self, policy):
        networkList = self.getNetworkList()
        if len(networkList) > 0:
            networkList.sort(key=lambda x: x["creation"], reverse=True)
            logMsg("Continue training of an existing network", networkList[0])
            networkId = networkList[0]["id"]

            networkData = self.downloadNetwork(networkId)

            policy.load(networkData)

            logMsg("Network %s loaded" % policy.getUUID())
        

    def getNetworkList(self):
        return requestJson(self.commandHost + "/api/networks/list/" + self.runId, self.secret)

    def uploadNetwork(self, policy):
        newPolicyEncoded = encodeToBson(policy.store())
        self.uploadEncodedNetwork(policy.getUUID(), newPolicyEncoded)
   
    def uploadEncodedNetwork(self, policyUUID, newPolicyEncoded):
        postBytes(self.commandHost + "/api/networks/" + self.runId + "/" + policyUUID, self.secret, newPolicyEncoded)
        logMsg("Network uploaded successfully!")

    def downloadNetwork(self, networkId):
        return decodeFromBson(requestBytes(self.commandHost + "/api/networks/download/" + networkId, self.secret))
