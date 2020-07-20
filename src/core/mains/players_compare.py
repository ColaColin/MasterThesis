# inputs:
# command, run, network
# players (just hardcoded...)

# every player plays games against every other player, N as the first player, N as the second player. (probably N = 200, we'll see)
# games are played using the same random-exploration as in normal self-play.

# outputs:
# result % win rates

# hard coded players to be used.


import setproctitle
from core.mains.mlconfsetup import loadMlConfig
import sys
from utils.prints import logMsg, setLoggingEnabled

from core.training.NetworkApi import NetworkApi

from core.mains.remote_evaluator import getRunConfig

# for run 395ba2f7-4e40-4cbd-ae57-19087d344e25, iteration 25 (network is 67bbb2dd-f3d1-477c-a3dd-a22bd7f74c3d)
players = {
    "extended": {
        "cpuct": 1.545,
        "drawValue": 0.6913,
        "fpu": 0.8545,
        "alphaBase": 20.38
    },
    "evolved": {
        "cpuct": 0.5590,
        "fpu": 0.1685,
        "alphaBase": 3.6369,
        "drawValue": 0.1037
    }
}

# for run 325d9f51-97d2-48ab-8999-25f2583979ba, iteration 0 random network
# players = {
#     "extended": {
#         "cpuct": 1.545,
#         "drawValue": 0.6913,
#         "fpu": 0.8545,
#         "alphaBase": 20.38
#     },
#     "hyperopt": {
#         "cpuct": 0.08295,
#         "drawValue": 0.106,
#         "fpu": 0.9995,
#         "alphaBase": 20.38
#     }
# }

# for run 325d9f51-97d2-48ab-8999-25f2583979ba, iteration 1 (network is 8b0d0678-1fce-4f9e-a286-8ba7fc9b8769)
# players = {
#     "evolved": {
#         "cpuct": 0.5328,
#         "drawValue": 0.8753,
#         "fpu": 0.9912,
#         "alphaBase": 20.38
#     },
#     "extended": {
#         "cpuct": 1.545,
#         "drawValue": 0.6913,
#         "fpu": 0.8545,
#         "alphaBase": 20.38
#     },
#     "hyperopt": {
#         "cpuct": 0.8162,
#         "drawValue": 1,
#         "fpu": 0.5255,
#         "alphaBase": 20.38
#     }
# }

# for run 325d9f51-97d2-48ab-8999-25f2583979ba, iteration 23 (network is 85618306-f76a-4548-b3bc-bf3b0ae46d74)
# players = {
#     "evolved": {
#         "cpuct": 0.2925,
#         "drawValue": 0.4728,
#         "fpu": 0.1411,
#         "alphaBase": 20.38
#     },
#     "extended": {
#         "cpuct": 1.545,
#         "drawValue": 0.6913,
#         "fpu": 0.8545,
#         "alphaBase": 20.38
#     },
#     "hyperopt": {
#         "cpuct": 0.9045,
#         "drawValue": 0.4498,
#         "fpu": 0.000977,
#         "alphaBase": 20.38
#     }
# }

def comparePlayers(player1, player2, games=1000):
    hasArgs = ("--secret" in sys.argv) and ("--command" in sys.argv) and ("--run" in sys.argv)

    if not hasArgs:
        raise Exception("You need to provide arguments for the hyperopt_players: --secret <server password> and --command <command server host> --run <run id>!")

    secret = sys.argv[sys.argv.index("--secret")+1]
    commandHost = sys.argv[sys.argv.index("--command")+1]
    runId = sys.argv[sys.argv.index("--run")+1]
    if "--network" in sys.argv:
        networkId = sys.argv[sys.argv.index("--network")+1]
    else:
        networkId = None

    networks = NetworkApi(noRun=True)

    with getRunConfig(runId, commandHost, secret) as temp:
        core = loadMlConfig(temp.name)

    policy = core.worker.policy(recursive=True)
    if networkId is not None:
        unpackedNetwork = networks.downloadNetwork(networkId)
        policy.load(unpackedNetwork)
        logMsg("Loaded policy id %s" % policy.getUUID())
    else:
        logMsg("Using random policy!")

    policyIterator = core.worker.policyIterator(recursive=True)
    moveDecider = core.worker.moveDecider(recursive=True)
    initialState = core.worker.initialState(recursive=True)

    playingGames = []
    for gidx in range(games):
        hdict = dict()
        if gidx >= games // 2:
            hdict[1] = player1
            hdict[2] = player2
        else:
            hdict[1] = player2
            hdict[2] = player1

        playingGames.append([initialState, hdict])

    logMsg("Beginning to play games with players:", player1, player2)

    turn = 0

    while True:
        turn += 1

        remainingGames = list(filter(lambda x: not x[0].hasEnded(), playingGames))

        logMsg("Turn %i, remaining games: %i" % (turn, len(remainingGames)))

        if len(remainingGames) == 0:
            break

        iterators = list(map(lambda x: policyIterator.createIterator(x[0], x[1]), remainingGames))
        policyIterator.iteratePolicyEx(policy, iterators)
        iterationResults = [ix.getResult() for ix in iterators]
        movesToPlay = list(map(lambda x: moveDecider.decideMove(x[0][0], x[1][0], x[1][1]), zip(remainingGames, iterationResults)))
        for move, game in zip(movesToPlay, remainingGames):
            game[0] = game[0].playMove(move)

    wins1 = 0
    wins2 = 0
    draws = 0
    for state, playerDict in playingGames:
        winnerNumber = state.getWinnerNumber()
        if winnerNumber > 0:
            winnerDict = playerDict[winnerNumber]
            if winnerDict == player1:
                wins1 += 1
            elif winnerDict == player2:
                wins2 += 1
        else:
            draws += 1

    return wins1, wins2, draws

if __name__ == "__main__":
    setproctitle.setproctitle("x0_player_bayesian_opt")
    setLoggingEnabled(True)
    playerNames = list(dict.keys(players))

    resultDict = dict()

    for p1Idx in range(len(playerNames)):
        for p2Idx in range(p1Idx+1, len(playerNames)):
            p1Name = playerNames[p1Idx]
            p2Name = playerNames[p2Idx]
            p1Dict = players[p1Name]
            p2Dict = players[p2Name]
    
            wins1, wins2, draws = comparePlayers(p1Dict, p2Dict)

            resultDict[(p1Name, p2Name)] = (wins1, wins2, draws)

            print("%s (%i)" % (p1Name, wins1), "vs", "%s, (%i)" % (p2Name, wins2), "Draws:", draws)
    
    print(resultDict)
