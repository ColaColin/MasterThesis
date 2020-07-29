# generate diagrams and stuff from data on the server
# just hack in the right values and run the script...

from utils.req import requestJson
import setproctitle

from utils.prints import logMsg, setLoggingEnabled

import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

# up to what hour of cost to display data
cutRight = 40
yLow = 84
yHigh = 92.5

command = "https://x0.cclausen.eu"

# plot_name = "Auxiliary features: Auxiliary network training costs are an issue"
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "move1, full costs": {
#         "color": (0, 1, 0),
#         "extraCost": 7200,
#         "runs": [
#             "4564200d-3a4c-425d-b54a-ee8f4ea9d998"
#         ]
#     },
#     "move1, ignore init cost": {
#         "color": (0, 0, 1),
#         "extraCost": 0,
#         "runs": [
#             "4564200d-3a4c-425d-b54a-ee8f4ea9d998"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/preCostCompare.eps"

plot_name = "Auxiliary features: Random network vs Trained network"
groups = {
    "extended": {
        "color": (1,0,0),
        "runs": [
            "7d675f5e-0926-43f9-b508-a55b06a42b2c",
            "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
            "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
            "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
            "658f11a2-a862-418e-a3b5-32d145d3dbdf"
        ]
    },
    "move1, trained": {
        "color": (0, 1, 0),
        "extraCost": 7200,
        "runs": [
            "4564200d-3a4c-425d-b54a-ee8f4ea9d998"
        ]
    },
    "move1, trained, ignore init cost": {
        "color": (1, 0.5, 0.5),
        "extraCost": 0,
        "runs": [
            "4564200d-3a4c-425d-b54a-ee8f4ea9d998"
        ]
    },
    "move1, random": {
        "color": (0, 0, 1),
        "extraCost": 0,
        "runs": [
            "782e0549-06b5-4157-a3c6-ce6954140fe3"
        ]
    }
}
img_output = "/ImbaKeks/git/MasterThesis/Write/images/rndVsTrainedAux.eps"



# plot_name = "Auxiliary features"
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "move1": {
#         "color": (0, 1, 0),
#         "extraCost": 7200,
#         "runs": [
#             "4564200d-3a4c-425d-b54a-ee8f4ea9d998"
#         ]
#     },
#     "move0, win0": {
#         "color": (0, 0, 1),
#         "extraCost": 7200,
#         "runs": [
#             "0e80434c-ac21-4e6c-b89c-67102e7d472c"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/auxiliary_attempt1.eps"

# plot_name = "Baselines, hard dataset"
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "base": {
#         "color": (0,0,1),
#         "runs": [
#             "12c31999-e8a9-4a52-b017-796e64b05f8a",
#             "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
#             "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
#             "bdf69def-4476-43fc-934d-115a7d895d6e",
#             "1edc288e-df3e-47c1-b9ce-52ab0045404a"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/baseline_ex.eps"

# plot_name = "Baselines, easy dataset"
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "base": {
#         "color": (0,0,1),
#         "runs": [
#             "12c31999-e8a9-4a52-b017-796e64b05f8a",
#             "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
#             "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
#             "bdf69def-4476-43fc-934d-115a7d895d6e",
#             "1edc288e-df3e-47c1-b9ce-52ab0045404a"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/baseline_ex_easy_dataset.eps"
# command = "http://127.0.0.1:8042"
# yHigh = 97.5

# plot_name = "Deduplication with different weight factors"
# groups = {
#     "base": {
#         "color": (0,0,1),
#         "runs": [
#             "12c31999-e8a9-4a52-b017-796e64b05f8a",
#             "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
#             "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
#             "bdf69def-4476-43fc-934d-115a7d895d6e",
#             "1edc288e-df3e-47c1-b9ce-52ab0045404a"
#         ],
#         "onlyMean": True
#     },
#     "weight 0.2": {
#         "color": (1,0,0),
#         "runs": [
#             "505b834a-345a-4a1b-a67a-fa405b27d6e4"
#         ]
#     },
#     "weight 0.5": {
#         "color": (0,1,0),
#         "runs": [
#             "43a39db5-5eec-43d9-9e50-a945248a64e8"
#         ]
#     },
#     "weight 0.8": {
#         "color": (0,0.5,0.5),
#         "runs": [
#             "8b1900b0-d133-4435-baf5-6c35934ff94c"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/dedupe.eps"

# plot_name = "Cyclic learning rate"
# groups = {
#     "base": {
#         "color": (0,0,1),
#         "runs": [
#             "12c31999-e8a9-4a52-b017-796e64b05f8a",
#             "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
#             "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
#             "bdf69def-4476-43fc-934d-115a7d895d6e",
#             "1edc288e-df3e-47c1-b9ce-52ab0045404a"
#         ],
#         "onlyMean": True
#     },
#     "cyclic_lr": {
#         "color": (1,0,0),
#         "runs": [
#             "7d434f56-e7c0-4945-af3b-3abdb30f4fca"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/cyclic_results.eps"

# plot_name = "Slow training window"
# groups = {
#     "base": {
#         "color": (0,0,1),
#         "runs": [
#             "12c31999-e8a9-4a52-b017-796e64b05f8a",
#             "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
#             "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
#             "bdf69def-4476-43fc-934d-115a7d895d6e",
#             "1edc288e-df3e-47c1-b9ce-52ab0045404a"
#         ],
#         "onlyMean": True
#     },
#     "slow window": {
#         "color": (1,0,0),
#         "runs": [
#             "32bb62a4-5541-4c0c-af1d-e84c09dfdccc"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/slow_window.eps"

# plot_name = "Playout Caps"
# groups = {
#     "base": {
#         "color": (0,0,1),
#         "runs": [
#             "12c31999-e8a9-4a52-b017-796e64b05f8a",
#             "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
#             "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
#             "bdf69def-4476-43fc-934d-115a7d895d6e",
#             "1edc288e-df3e-47c1-b9ce-52ab0045404a"
#         ],
#         "onlyMean": True
#     },
#     "Playout Caps": {
#         "color": (1,0,0),
#         "runs": [
#             "0538a5d8-0706-4b90-b601-c0adbfd69cc6"
#         ]
#     },
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/playout_caps.eps"


# plot_name = "Predict the opponent's reply"
# groups = {
#     "base": {
#         "color": (0,0,1),
#         "runs": [
#             "12c31999-e8a9-4a52-b017-796e64b05f8a",
#             "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
#             "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
#             "bdf69def-4476-43fc-934d-115a7d895d6e",
#             "1edc288e-df3e-47c1-b9ce-52ab0045404a"
#         ],
#         "onlyMean": True
#     },
#     "Playout Caps": {
#         "color": (1,0,0),
#         "runs": [
#             "fd514ad3-35db-44e9-8768-76c5822dc09e"
#         ]
#     },
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/predict_reply.eps"


# plot_name = "Squeeze and Excite ResNet"
# groups = {
#     "base": {
#         "color": (0,0,1),
#         "runs": [
#             "12c31999-e8a9-4a52-b017-796e64b05f8a",
#             "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
#             "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
#             "bdf69def-4476-43fc-934d-115a7d895d6e",
#             "1edc288e-df3e-47c1-b9ce-52ab0045404a"
#         ],
#         "onlyMean": True
#     },
#     "Squeeze and Excite": {
#         "color": (1,0,0),
#         "runs": [
#             "f64aae6e-f094-47b5-aa0c-1201b324e939"
#         ]
#     },
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/sqnet.eps"


# plot_name = "Hyperparameter comparison"
# groups = {
#     "hyperopt1": {
#         "color": (1, 0, 0),
#         "runs": [
#             "aa4782ae-c162-4443-a290-41d7bb625d17",
#             "3d09cdce-4e69-4811-bda9-ad2985228230",
#             "fe764034-ba1f-457b-8329-b5904bb8f66c",
#             "3eca837f-4b4d-439e-b6e7-09b35edf3d5d",
#             "55efa347-a847-4241-847e-7497d2497713"
#         ]
#     },
#     "hyperopt2": {
#         "color": (0, 0, 1),
#         "runs": [
#             "1edc288e-df3e-47c1-b9ce-52ab0045404a",
#             "bdf69def-4476-43fc-934d-115a7d895d6e",
#             "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
#             "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
#             "12c31999-e8a9-4a52-b017-796e64b05f8a"
#         ]
#     },
#     "prevWork": {
#         "color": (0,1,0),
#         "runs": [
#             "65388799-c526-4870-b371-fb47e35e56af",
#             "583eae5c-94e8-4bdb-ac48-71bc418c5a51",
#             "bba7494a-c5f9-42bb-90ff-b93a91b5e74b",
#             "5350bdc8-8c4b-422b-8bfc-0435d2b6d45d",
#             "9733ab7c-7ebc-49eb-87db-1f03e0929d10"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/foobar.eps"

# plot_name = "Player evolution of key parameters"
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "evolution": {
#         "color": (0, 1, 0),
#         "runs": [
#             "325d9f51-97d2-48ab-8999-25f2583979ba"
#         ]
#     },
#     "extra": "diversity"
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/player_evolution_low_diversity.eps"

# plot_name = "Player evolution of key parameters"
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "evolve cpuct, fpu, drawValue": {
#         "color": (0, 1, 0),
#         "runs": [
#             "325d9f51-97d2-48ab-8999-25f2583979ba"
#         ]
#     },
#     "evolve kldgain": {
#         "color": (0, 0, 1),
#         "runs": [
#             "1a4c1c39-a812-4f82-9da4-17bf237baeb7"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/evolve_results.eps"

# plot_name = "Player evolution of key parameters and training data diversity"
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "evolve cpuct, fpu, drawValue": {
#         "color": (0, 1, 0),
#         "runs": [
#             "325d9f51-97d2-48ab-8999-25f2583979ba"
#         ]
#     },
#     "evolve kldgain": {
#         "color": (0, 0, 1),
#         "runs": [
#             "1a4c1c39-a812-4f82-9da4-17bf237baeb7"
#         ]
#     },
#     "extra": "diversity"
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/evolve_low_diversity.eps"

# plot_name = "Player evolution: Novelty search"
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "Optimize win chance": {
#         "color": (0, 1, 0),
#         "runs": [
#             "325d9f51-97d2-48ab-8999-25f2583979ba"
#         ]
#     },
#     "Optimize novel wins": {
#         "color": (0, 0, 1),
#         "runs": [
#             "395ba2f7-4e40-4cbd-ae57-19087d344e25"
#         ]
#     },
#     "extra": "diversity"
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/player_evolution_win_novelty.eps"

# plot_name = "Evolving player parameters for novelty search only."
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "novelty search": {
#         "color": (0, 1, 0),
#         "runs": [
#             "f8830ef7-0e14-4e0d-ae29-87378daf5b5f",
#             "d2e2917f-4ca3-4c13-89d0-ebdf2ca152e6",
#             "037fa6cc-4975-459d-9a84-98ce9eb1342d"
#         ]
#     },
#     "extra": "diversity"
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/pure_novelty_search.eps"


# plot_name = "Caching all MCTS results with the help of an MCTS evaluation service."
# groups = {
#     "extended": {
#         "color": (1,0,0),
#         "runs": [
#             "7d675f5e-0926-43f9-b508-a55b06a42b2c",
#             "5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86",
#             "b9336ccf-69e1-4ad4-8a5a-246e734d7a81",
#             "e2f7655f-94f4-4e58-9397-a3b8d11ef5d8",
#             "658f11a2-a862-418e-a3b5-32d145d3dbdf"
#         ]
#     },
#     "cached MCTS": {
#         "color": (0, 1, 0),
#         "runs": [
#             "1d182bb0-5b26-49fb-b2a9-4417322f76e5",
#             "d91dbba7-4363-4779-8a86-7a127977d9e4",
#             "e1500cbb-45ae-4e1a-a55a-8015fa414afd"
#         ]
#     }
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/cache_play.eps"


# plot_name = "Exploration by retrying different moves after losses."
# groups = {
#     "cached MCTS": {
#         "color": (1, 0, 0),
#         "runs": [
#             "1d182bb0-5b26-49fb-b2a9-4417322f76e5",
#             "d91dbba7-4363-4779-8a86-7a127977d9e4",
#             "e1500cbb-45ae-4e1a-a55a-8015fa414afd"
#         ]
#     },
#     "Retry after loss": {
#         "color": (0, 1, 0),
#         "runs": [
#             "e6135ef6-e360-47d7-b9bb-bfe91f3a341b"
#         ]
#     },
#     "extra": "diversity"
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/winp_tree.eps"

# plot_name = "Exploration by MCTS"
# groups = {
#     "cached MCTS baseline": {
#         "color": (1, 0, 0),
#         "runs": [
#             "1d182bb0-5b26-49fb-b2a9-4417322f76e5",
#             "d91dbba7-4363-4779-8a86-7a127977d9e4",
#             "e1500cbb-45ae-4e1a-a55a-8015fa414afd"
#         ]
#     },
#     "Exploration by MCTS, cpuct = 15": {
#         "color": (0, 1, 0),
#         "runs": [
#             "e5eb7ac2-3123-46bd-a79a-5026814a859c"
#         ]
#     },
#     "extra": "diversity"
# }
# img_output = "/ImbaKeks/git/MasterThesis/Write/images/mcts_tree_explore.eps"




def getMeanOf(lsts):
    maxLength = 0
    for lst in lsts:
        if len(lst) > maxLength:
            maxLength = len(lst)
    meanPoints = []

    for i in range(maxLength):
        cs = []
        for cost in lsts:
            if len(cost) > i:
                cs.append(cost[i])
        meanPoints.append(np.mean(cs))

    return meanPoints

def interpolateForY(xData, yData, yPoint):
    idx = 0
    while idx < len(yData) and yData[idx] < yPoint:
        idx += 1
    
    if idx == 0 or idx >= len(yData) or idx >= len(xData):
        return None

    xAfter = xData[idx]
    xBefore = xData[idx-1]

    yAfter = yData[idx]
    yBefore = yData[idx-1]

    assert yPoint <= yAfter and yPoint >= yBefore

    yRange = yAfter - yBefore
    linScale = (yPoint - yBefore) / yRange

    return xBefore + (xAfter - xBefore) * linScale

aFlip = 0

def annote_max(y,x, ax=None):
    xmax = x[np.argmax(y)]
    ymax = max(y)

    text= "cost={:.1f}, acc={:.1f}".format(xmax, ymax)

    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle")
    kw = dict(xycoords='data',textcoords="data",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")

    global aFlip

    if aFlip % 2 == 0:
        yOffset = -10
    else:
        yOffset = -5
    
    aFlip += 1

    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax + 5, ymax + yOffset), **kw)

def whitenColor(rgb, f):
    rgbv = np.array(rgb)
    whitev = np.array([1,1,1])

    return rgbv * f + (1 - f) * whitev


def plotGroup(name, fig, ax, ax2, extraStats = None):
    runs = groups[name]["runs"]
    color = groups[name]["color"]
    extraCosts = 0
    if "extraCost" in groups[name]:
        extraCosts = groups[name]["extraCost"] / 3600.0

    datas = []

    extras = []

    for run in runs:
        data = requestJson(command + "/costs/" + run, "")
        costs = []
        accs = []
        for d in data:
            cost = d["cost"]
            acc = d["acc_mcts_moves"]
            if cost is not None and acc is not None:
                costs.append(cost + extraCosts)
                accs.append(acc)
        datas.append((costs, accs))

        if extraStats is not None:
            edata = requestJson(command + "/tables/" + extraStats + "/" + run, "")
            costs = []
            extraVals = []
            for ed in edata:
                costs.append(ed["cost"] + extraCosts)
                extraVals.append(ed[extraStats])
            extras.append((costs, extraVals))

    if len(datas) > 1:
        if not "onlyMean" in groups[name] or not groups[name]["onlyMean"]:
            for cost, acc in datas:
                ax.plot(cost, acc, color=whitenColor(color, 0.3), linewidth=1)

        meanX, meanY = meanInterpolatedLine(datas)
        ax.plot(meanY, meanX, label=name, color=color+(1,), linewidth=2)

        #annote_max(meanX, meanY, ax)

        mostLeft = meanY[0] * 0.99
        mostRight = meanY[-1] * 1.01
    elif len(datas) == 1:
        mostLeft = datas[0][0][0] * 0.99
        mostRight = datas[0][0][0] * 1.01
        ax.plot(datas[0][0], datas[0][1], color=whitenColor(color, 0.9), linewidth=1, label=name)

        #annote_max(datas[0][0], datas[0][1], ax)

    if extraStats is not None:
        if len(extras) > 1:
            for cost, acc in extras:
                ax2.plot(cost, acc, "--", color=whitenColor(color, 0.3), linewidth=1)

            meanX, meanY = meanInterpolatedLine(extras)
            ax2.plot(meanY, meanX, "--", label=name, color=color+(1,), linewidth=2)
        elif len(extras) == 1:
            ax2.plot(extras[0][0], extras[0][1], "--", label=name, color=whitenColor(color, 0.9), linewidth=1)

    return mostLeft, mostRight

def meanInterpolatedLine(datas):
    y = getMeanOf(list(map(lambda x: x[0], datas)))

    xPoints = []
    yPoints = []

    for yP in y:
        xR = []
        ok = True
        for yData, xData in datas:
            xLin = interpolateForY(xData, yData, yP)
            if xLin is not None:
                xR.append(xLin)
            else:
                ok = False
        if ok:
            xPoints.append(np.mean(xR))
            yPoints.append(yP)

    return xPoints, yPoints

if __name__ == "__main__":
    setLoggingEnabled(True)
    setproctitle.setproctitle("x0_generate_tables")

    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(1,1,1)

    mostLeft = None
    mostRight = None

    if "extra" in groups:
        extraStats = groups["extra"]
        ax2 = ax.twinx()
        ax2.set_ylabel(extraStats)
    else:
        extraStats = None
        ax2 = None

    for groupName in groups:
        if groupName == "extra":
            continue
        mL, mR = plotGroup(groupName, fig, ax, ax2, extraStats)

        if mostLeft is None or mostLeft > mL:
            mostLeft = mL

        if mostRight is None or mostRight < mR:
            mostRight = mR

    if mostRight > cutRight:
        mostRight = cutRight


    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.5, color="black")
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="black")

    plt.xlim([mostLeft, mostRight])

    ax.set_ylim([yLow, yHigh])

    ax.legend(loc="lower left" if ax2 is not None else "lower right", fancybox=True, title="Accuracy")
    if ax2 is not None:
        ax2.legend(loc="lower right", fancybox=True, title="Diversity")

    plt.title(plot_name)

    ax.set_ylabel("MCTS accuracy %")
    plt.xlabel("Estimated cost in hours")

    plt.savefig(img_output, bbox_inches="tight", format="eps", dpi=300)

    plt.show()