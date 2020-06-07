import abc

from core.playing.SelfPlayWorker import GameReporter, PolicyUpdater
from utils.prints import logMsg
import time

import sys
import requests
from utils.bsonHelp.bsonHelp import encodeToBson, decodeFromBson

from utils.misc import readFileUnderPath, openJsonFile

from datetime import datetime

import subprocess
import os

class DistributedReporter(GameReporter, metaclass=abc.ABCMeta):
    def __init__(self, packageSize = 1000):
        """
        Requires some configuration parameters to be present in the arguments to python:
        --command <command server host>
        --secret <server api password>
        --run <run-uuid>
        --worker <worker-name>

        @param packageSize: Determines how many game states are included in a single network call to do a report. Default: 1000
        """
        self.packageSize = packageSize

        hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--worker" in sys.argv) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for the distributed worker: --secret <server password>, --run <uuid>, --worker <name> and --command <command server host>!")

        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.run = sys.argv[sys.argv.index("--run")+1]
        self.workerName = sys.argv[sys.argv.index("--worker")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]

        self.queue = []
        self.waitTill = time.monotonic() - 1


    def reportGame(self, reports):
        for report in reports:
            self.queue.append(report)

        if len(self.queue) > self.packageSize:
            if self.waitTill > time.monotonic():
                return

            pending = self.queue[:self.packageSize]

            try:
                response = requests.post(url=self.commandHost + "/api/state/" + self.workerName + "/" + self.run,
                    data=encodeToBson(pending),
                    headers={"secret": self.secret})
                response.raise_for_status()
                reportId = response.json()

                logMsg("Reported ", len(pending), "states to the server, they were assigned report id ", reportId)

                self.queue = self.queue[self.packageSize:]
            except Exception as error:
                logMsg("Could not report states due to an error. Will try again soon!", error)
                self.waitTill = time.monotonic() + 60

class DistributedNetworkUpdater2(PolicyUpdater, metaclass=abc.ABCMeta):
    """
    Version of the distributed networks updater that uses an extra networks_downloader process to handle network downloading.
    That process is stared once by every worker, but it exits if it detects another one was already started before, so there is only one downloading process.
    """

    def __init__(self, storage):
        self.lastNetworkCheck = -999
        self.checkInterval = 4
        self.storage = storage

        hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--worker" in sys.argv) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for the distributed worker: --secret <server password>, --run <uuid>, --worker <name> and --command <command server host>!")

        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.run = sys.argv[sys.argv.index("--run")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]

        self.downloader = subprocess.Popen(["python", "-m", "core.mains.networks_downloader", "--path", self.storage, "--secret", self.secret, "--command", self.commandHost, "--run", self.run])

    def update(self, policy):
        if time.monotonic() - self.lastNetworkCheck > self.checkInterval:
            downloader.poll()
            
            self.lastNetworkCheck = time.monotonic()

            npath = os.path.join(self.storage, "networks.json")

            if os.path.exists(npath):
                try:
                    networks = openJsonFile(npath)
                    if len(networks) > 0:
                        networks.sort(key=lambda n: n["creation"])
                        bestNetwork = networks[-1]
                        if bestNetwork["id"] != policy.getUUID():
                            logMsg("New network found created at UTC", datetime.utcfromtimestamp(bestNetwork["creation"] / 1000).strftime('%Y-%m-%d %H:%M:%S'))
                            policy.load(decodeFromBson(readFileUnderPath(os.path.join(self.storage, bestNetwork["id"]))))
                            logMsg("Policy replaced, now working with policy ", policy.getUUID())

                except Exception as error:
                    logMsg("Failed to check for a new network", error)

        return policy


class DistributedNetworkUpdater(PolicyUpdater, metaclass=abc.ABCMeta):

    def __init__(self, checkInterval = 15):
        """
        Requires some configuration parameters to be present in the arguments to python (a subset of what is needed for DistributedReporter)
        --command <command server host>
        --secret <server api password>
        --run <run-uuid>

        @param checkInterval: How many seconds to wait before checking if a newer network is available. Default 15
        """
        self.lastNetworkCheck = -999
        self.checkInterval = checkInterval

        hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--worker" in sys.argv) and ("--command" in sys.argv)

        if not hasArgs:
            raise Exception("You need to provide arguments for the distributed worker: --secret <server password>, --run <uuid>, --worker <name> and --command <command server host>!")

        self.secret = sys.argv[sys.argv.index("--secret")+1]
        self.run = sys.argv[sys.argv.index("--run")+1]
        self.commandHost = sys.argv[sys.argv.index("--command")+1]

    def update(self, policy):
        if time.monotonic() - self.lastNetworkCheck > self.checkInterval:
            self.lastNetworkCheck = time.monotonic()

            try:
                response = requests.get(url=self.commandHost + "/api/networks/list/" + self.run,
                    headers={"secret": self.secret})
                response.raise_for_status()

                list = response.json()
                list.sort(key=lambda n: n["creation"])

                if len(list) > 0:
                    bestNetwork = list[-1]
                    if bestNetwork["id"] != policy.getUUID():
                        logMsg("New network found created at UTC", datetime.utcfromtimestamp(bestNetwork["creation"] / 1000).strftime('%Y-%m-%d %H:%M:%S'), "\nDownloading...")

                        response = requests.get(url=self.commandHost + "/api/networks/download/" + bestNetwork["id"],
                            headers={"secret": self.secret}, stream=True)
                        response.raise_for_status()

                        logMsg("Download completed!")

                        policy.load(decodeFromBson(response.raw.data))

                        logMsg("Policy replaced, now working with policy ", policy.getUUID())

            except Exception as error:
                logMsg("Could not query for new network this time, will try again soon!", error)

        return policy