

import setproctitle

from utils.prints import logMsg, setLoggingEnabled

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events

import time
import os

import subprocess

import ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True

import json

from core.mains.mlconfsetup import loadMlConfig

import numpy as np
from core.solved.PolicyTester import PolicyIteratorPlayer, DatasetPolicyTester2

from impls.selfplay.movedeciders import TemperatureMoveDecider

current_milli_time = lambda: int(round(time.time() * 1000))

zeroTime = current_milli_time()

sPerIteration = 1 * 60 * 60
baseWorkDir = "/ImbaKeks/runs/hyperoptZ"

# Run 1
# |  1        |  75.8     |  14.26    |  4.392    |  0.000114 |  0.3023   |
# |  2        |  78.97    |  6.962    |  0.7809   |  0.1863   |  0.3456   |
# |  3        |  79.35    |  6.951    |  1.03     |  0.1371   |  0.2909   |

# Run 2
#|  1        |  76.29    |  14.26    |  4.392    |  0.000114 |  0.3023   |
#|  2        |  78.73    |  6.962    |  0.7809   |  0.1863   |  0.3456   |
#|  3        |  79.79    |  6.951    |  1.03     |  0.1371   |  0.2909   |

# Run 3
#|  1        |  76.82    |  14.26    |  4.392    |  0.000114 |  0.3023   |
#|  2        |  80.07    |  6.962    |  0.7809   |  0.1863   |  0.3456   |
#|  3        |  78.5     |  6.951    |  1.03     |  0.1371   |  0.2909   |

def writeConfig(outDir, blocks, filters, extraFilters, nodes, cpuct, rootNoise, drawValue, explorationPlyCount, fpu, lr, wd, windowSize, reportsPerIteration, alphaBase, epochs):
    template = "confs/hyperopt.yaml"

    with open(template) as f:
        config = yaml.load(f)
    
    config["resnet"]["blocks"] = int(round(blocks))
    config["resnet"]["filters"] = int(round(filters))
    config["resnet"]["extraHeadFilters"] = int(round(extraFilters))

    config["bestMcts"]["expansions"] = int(round(nodes))
    config["mcts"]["expansions"] = int(round(nodes))

    config["bestMcts"]["cpuct"] = float(cpuct)
    config["mcts"]["cpuct"] = float(cpuct)

    config["bestMcts"]["rootNoise"] = float(0.001)
    config["mcts"]["rootNoise"] = float(rootNoise)

    config["bestMcts"]["drawValue"] = float(drawValue)
    config["mcts"]["drawValue"] = float(drawValue)

    config["bestMcts"]["alphaBase"] = float(alphaBase)
    config["mcts"]["alphaBase"] = float(alphaBase)

    config["bestMcts"]["fpu"] = float(fpu)
    config["mcts"]["fpu"] = float(fpu)

    config["tempDecider"]["explorationPlyCount"] = int(round(explorationPlyCount))

    config["optimizerArgs"]["lr"] = float(lr)
    config["optimizerArgs"]["weight_decay"] = float(wd)

    config["reporter"]["windowSize"] = int(round(windowSize))
    config["reporter"]["reportsPerIteration"] = int(round(reportsPerIteration))
    config["reporter"]["state"] = outDir
    
    config["updater"]["trainEpochs"] = int(round(epochs))
    config["updater"]["state"] = outDir

    path = os.path.join(outDir, "params.yaml")

    with open(path, "w") as f:
        yaml.dump(config, f)

    return path

def wdef(m, k, default):
    if k in m:
        return m[k]
    else:
        return default

def getScore(**kwargs):
    logMsg("Configuration under test:", kwargs)

    blocks = wdef(kwargs, "blocks", 5)
    filters = wdef(kwargs, "filters", 128)
    extraFilters = wdef(kwargs, "extraFilters", 32)
    nodes = wdef(kwargs, "nodes", 300)
    cpuct = wdef(kwargs, "cpuct", 4)
    rootNoise = wdef(kwargs, "rootNoise", 0.25)
    drawValue = wdef(kwargs, "drawValue", 0.5)
    explorationPlyCount = wdef(kwargs, "explorationPlyCount", 30)
    fpu = wdef(kwargs, "fpu", 0)
    lr = wdef(kwargs, "lr", 0.2)
    wd = wdef(kwargs, "wd", 0.0001)
    windowSize = wdef(kwargs, "windowSize", 100000)
    reportsPerIteration = wdef(kwargs, "reportsPerIteration", 10000)
    alphaBase = wdef(kwargs, "alphaBase", 10)
    epochs = wdef(kwargs, "epochs", 1)

    # the experiment will work in <baseWorkDir>/<ms>/
    workDir = os.path.join(baseWorkDir, str(current_milli_time() - zeroTime))
    os.mkdir(workDir)

    # generate configuration and store in the workDir
    configPath = writeConfig(workDir, blocks, filters, extraFilters, nodes, cpuct, rootNoise, drawValue, explorationPlyCount, fpu, lr, wd, windowSize, reportsPerIteration, alphaBase, epochs);

    # spawn localselfplay process that uses the workDir, producing networks that reside in that directory (modified storeState of SingleProcessUpdater for that).
    # the process will be killed after the given number of hours
    startTime = time.time()

    try:
        process = None
        with open(os.path.join(workDir, "logfile.txt"), "a+") as f:
            process = subprocess.Popen(["python", "-m", "core.mains.main", configPath], stdout=f, stderr=subprocess.STDOUT)

            while time.time() - startTime < sPerIteration:
                if process.poll() is not None:
                    raise Exception("worker died?")
                time.sleep(0.1)

            process.kill()
            process = None
    finally:
        if process is not None:
            process.kill()
    
    # evaluate results, loop through all the generated networks (file ending .hyperopt) and find the best one
    networks = list(map(lambda x: os.path.join(workDir, x), filter(lambda x: x.endswith(".hyperopt.npy"), os.listdir(workDir))))
    bestAccuracy = 0

    for network in networks:
        core = loadMlConfig(configPath)
        policy = core.main.policy(recursive=True)
        policy.load(np.load(network))
        policyIterator = core.bestMcts(recursive=True)
        moveDecider = TemperatureMoveDecider(-1)
        initialState = core.main.initialState(recursive=True)
        policyPlayer = PolicyIteratorPlayer(policy, policyIterator, None, moveDecider, 128)
        bestTester = DatasetPolicyTester2(policyPlayer, "datasets/connect4/testset.txt.zip", initialState)
        bestResult, _ = bestTester.main()
        if bestResult > bestAccuracy:
            bestAccuracy = bestResult

    return bestAccuracy

class CombinedLogger():
    def __init__(self, path):
        self.json = JSONLogger(path)
        self.console = ScreenLogger()
    
    def update(self, event, instance):
        self.json.update(event, instance)
        self.console.update(event, instance)

if __name__ == "__main__":
    setproctitle.setproctitle("x0_hyperopt")
    setLoggingEnabled(False)

    pbounds = {
        #'blocks': (2, 4),
        #'filters': (32, 64),
        #'extraFilters': (4, 16),
        #'nodes': (20, 40),
        'cpuct': (0.25, 6),
        #'rootNoise': (0.01, 0.5),
        'drawValue': (0, 1),
        #'explorationPlyCount': (10, 40),
        'fpu': (0, 1),
        #'lr': (0.0005, 0.005),
        #'wd': (0.00005, 0.001),
        #'windowSize': (100000, 250000),
        #'reportsPerIteration': (10000, 42000),
        'alphaBase': (3, 30),
        #'epochs': (1, 3)
    }

    optimizer = BayesianOptimization(
        f = getScore,
        pbounds = pbounds,
        random_state= 1
    );

    logger = CombinedLogger(os.path.join(baseWorkDir, "logs.json"))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=2,
        n_iter=1
    )

    print(optimizer.max)

    with open(os.path.join(baseWorkDir, "best.json"), "w") as f:
        json.dump(optimizer.max, f)

    