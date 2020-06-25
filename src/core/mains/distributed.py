from core.mains.mlconfsetup import mlConfigBasedMain
import sys
from utils.req import requestJson
import json
import os
from utils.prints import logMsg, setLoggingEnabled
import setproctitle



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
# --worker <worker-name> A self-play worker for linear or league self play.
# --eval <worker-name> An evaluation worker meant for MCTS Tree self play.
# --training spawns the trainer instead of the playing-worker

# example calls
# python -m core.mains.distributed --command 'http://127.0.0.1:8042' --secret 42 --run 'c8e187a0-de60-4251-b985-9b3464b831dd' --worker test1
# python -m core.mains.distributed --command https://x0.cclausen.eu --secret 42 --run '4cdf1719-ed90-455b-b9a0-ac3a5c7fbab5' --worker test1

# python -m core.mains.distributed --command https://x0.cclausen.eu --secret 42 --run '4cdf1719-ed90-455b-b9a0-ac3a5c7fbab5' --eval test1

# for the trainer it might be necessary to increase the open file limit:
# ulimit -n 300000; python -m ...

if __name__ == "__main__":
    setLoggingEnabled(True)

    setproctitle.setproctitle("x0_distributed_setup")

    hasArgs = ("--secret" in sys.argv) and ("--run" in sys.argv) and ("--command" in sys.argv)

    if not hasArgs:
        raise Exception("You need to provide arguments for the distributed worker: --secret <server password>, --run <uuid> and --command <command server host>!")

    secret = sys.argv[sys.argv.index("--secret")+1]
    run = sys.argv[sys.argv.index("--run")+1]
    commandHost = sys.argv[sys.argv.index("--command")+1]

    runConfig = requestJson(commandHost + "/api/runs/" + run, secret)

    if "--fconfig" in sys.argv:
        # meant for frametime evaluator, not for general use.
        cfgPath = sys.argv[sys.argv.index("--fconfig") + 1]
        logMsg("Forced to use local file configuration %s" % cfgPath)
    else:
        cfgPath = os.path.join(os.getcwd(), "downloaded_config.yaml")
        with open(cfgPath, "w") as f:
            f.write(runConfig["config"])
        logMsg("Downloaded configuration to", cfgPath)
        logMsg("Running configuration")

    core = mlConfigBasedMain(cfgPath)

    if "--training" in sys.argv:
        setproctitle.setproctitle("x0_trainer_" + runConfig["name"])
        core.trainer(recursive=True).main()
    elif "--eval" in sys.argv:
        setproctitle.setproctitle("x0_evaluator_" + sys.argv[sys.argv.index("--eval")+1] + "_" + runConfig["name"])
        core.evalWorker(recursive=True).main()
    else:
        if "--worker" in sys.argv:
            setproctitle.setproctitle("x0_worker_" + sys.argv[sys.argv.index("--worker")+1] + "_" + runConfig["name"])
        core.worker(recursive=True).main()


