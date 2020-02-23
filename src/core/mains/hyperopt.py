

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
from core.solved.PolicyTester import PolicyIteratorPlayer, DatasetPolicyTester

from impls.selfplay.movedeciders import TemperatureMoveDecider

current_milli_time = lambda: int(round(time.time() * 1000))

zeroTime = current_milli_time()

sPerIteration = 0.2 * 60 * 60
baseWorkDir = "/ImbaKeks/runs/hyperopt/A/"

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

def getScore(blocks, filters, extraFilters, nodes, cpuct, rootNoise, drawValue, explorationPlyCount, fpu, lr, wd, windowSize, reportsPerIteration, alphaBase, epochs):
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
        bestTester = DatasetPolicyTester(policyPlayer, "datasets/connect4/best_small.dataset", initialState, "shell", 128)
        bestResult = bestTester.main()
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
    setLoggingEnabled(True)

    pbounds = {
        'blocks': (2, 4),
        'filters': (32, 64),
        'extraFilters': (4, 16),
        'nodes': (20, 40),
        'cpuct': (0.5, 6),
        'rootNoise': (0.01, 0.5),
        'drawValue': (0, 1),
        'explorationPlyCount': (10, 40),
        'fpu': (0, 1),
        'lr': (0.0005, 0.005),
        'wd': (0.00005, 0.001),
        'windowSize': (100000, 250000),
        'reportsPerIteration': (10000, 42000),
        'alphaBase': (3, 30),
        'epochs': (1, 3)
    }

    optimizer = BayesianOptimization(
        f = getScore,
        pbounds = pbounds,
        random_state= 1
    );

    logger = CombinedLogger(os.path.join(baseWorkDir, "logs.json"))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=5,
        n_iter=25
    )

    print(optimizer.max)

    with open(os.path.join(baseWorkDir, "best.json"), "w") as f:
        json.dump(optimizer.max, f)

    