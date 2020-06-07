import setproctitle
import time

from utils.prints import logMsg, setLoggingEnabled
from utils.req import requestJson, requestBytes
from utils.misc import openJsonFile, writeJsonFile, storeFileUnderPath

import sys
import os
import shutil

def loopNetworksDownload(storage):
    secret = sys.argv[sys.argv.index("--secret")+1]
    run = sys.argv[sys.argv.index("--run")+1]
    commandHost = sys.argv[sys.argv.index("--command")+1]

    while True:

        #1 get list of existing networks
        #2 for every network in the list, check if the file exists, if not download the network into the file
        #3 store the new list of networks

        networks = requestJson(commandHost + "/api/networks/list/" + run, secret)
        
        for network in networks:
            spath = os.path.join(storage, network["id"])
            if not os.path.exists(spath):
                netbytes = requestBytes(commandHost + "/api/networks/download/" + network["id"], secret)
                storeFileUnderPath(spath, netbytes)
                logMsg("Downloaded a new network to %s" % spath)

        writeJsonFile(os.path.join(storage, "networks.json"), networks)

        time.sleep(2)

if __name__ == "__main__":
    setproctitle.setproctitle("x0_networks_downloader")
    setLoggingEnabled(True)

    hasArgs = ("--path" in sys.argv) and ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--command" in sys.argv)

    if not hasArgs:
        raise Exception("Missing args for networks downloader!")

    storagePath = sys.argv[sys.argv.index("--path") + 1]

    try:
        os.makedirs(storagePath, exist_ok=False)
    except FileExistsError:
        logMsg("Storage already exists, there must be another downloader already running, exiting...")
        exit(0)

    try:
        logMsg("Starting networks downloader, storage in %s" % storagePath)
        writeJsonFile(os.path.join(storagePath, "networks.json"), [])
        loopNetworksDownload(storagePath)
    finally:
        shutil.rmtree(storagePath)

    