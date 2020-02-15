from core.mains.mlconfsetup import mlConfigBasedMain
import sys
import requests
import json
import os
from utils.prints import logMsg, setLoggingEnabled

# loader for distributed workers
# given some command line parameters
# that define where the command server is
# talk to that server to download a configuration
# then run that configuration

# Requires some configuration parameters to be present in the arguments to python:
# --command <command server host>
# --secret <server api password>
# --run <run-uuid>
# Not needed by this script directly, but in a distributed setting you likely use the DistributedReporter, which additionally needs:
# --worker <worker-name>
# --training spawns the trainer instead of the playing-worker

# example call
# python -m core.mains.distributed --command 'http://127.0.0.1:8042' --secret 42 --run 'c8e187a0-de60-4251-b985-9b3464b831dd' --worker test1

if __name__ == "__main__":
    setLoggingEnabled(True)

    hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--command" in sys.argv)

    if not hasArgs:
        raise Exception("You need to provide arguments for the distributed worker: --secret <server password>, --run <uuid> and --command <command server host>!")

    secret = sys.argv[sys.argv.index("--secret")+1]
    run = sys.argv[sys.argv.index("--run")+1]
    commandHost = sys.argv[sys.argv.index("--command")+1]

    response = requests.get(commandHost + "/api/runs/" + run, headers={"secret": secret})
    response.raise_for_status()

    runConfig = response.json()

    cfgPath = os.path.join(os.getcwd(), "downloaded_config.yaml")

    with open(cfgPath, "w") as f:
        f.write(runConfig["config"])
    
    logMsg("Downloaded configuration to", cfgPath)
    logMsg("Running configuration")

    core = mlConfigBasedMain(cfgPath)
    if "--training" in sys.argv:
        core.trainer(recursive=True).main()
    else:
        core.worker(recursive=True).main()


