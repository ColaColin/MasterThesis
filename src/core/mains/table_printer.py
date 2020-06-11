# generate diagrams and stuff from data on the server
# just hack in the right values and run the script...

from utils.req import requestJson
import setproctitle

from utils.prints import logMsg, setLoggingEnabled

import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

#plot_name = "Baselines"
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


plot_name = "Hyperparameter comparison"
groups = {
    "hyperopt1": {
        "color": (1, 0, 0),
        "runs": [
            "aa4782ae-c162-4443-a290-41d7bb625d17",
            "3d09cdce-4e69-4811-bda9-ad2985228230",
            "fe764034-ba1f-457b-8329-b5904bb8f66c",
            "3eca837f-4b4d-439e-b6e7-09b35edf3d5d",
            "55efa347-a847-4241-847e-7497d2497713"
        ]
    },
    "hyperopt2": {
        "color": (0, 0, 1),
        "runs": [
            "1edc288e-df3e-47c1-b9ce-52ab0045404a",
            "bdf69def-4476-43fc-934d-115a7d895d6e",
            "59e19f40-c4f0-46e9-97f8-5a2b423ef7fc",
            "45d2087b-04f9-49ca-b2d9-5c8736da86b5",
            "12c31999-e8a9-4a52-b017-796e64b05f8a"
        ]
    },
    "prevWork": {
        "color": (0,1,0),
        "runs": [
            "65388799-c526-4870-b371-fb47e35e56af",
            "583eae5c-94e8-4bdb-ac48-71bc418c5a51",
            "bba7494a-c5f9-42bb-90ff-b93a91b5e74b",
            "5350bdc8-8c4b-422b-8bfc-0435d2b6d45d",
            "9733ab7c-7ebc-49eb-87db-1f03e0929d10"
        ]
    }
}

# up to what hour of cost to display data
cutRight = 40
yLow = 84
yHigh = 92.5

command = "https://x0.cclausen.eu"
#command = "http://127.0.0.1:8042"
img_output = "/ImbaKeks/git/MasterThesis/Write/images/hyper_compare.eps"

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


def plotGroup(name, fig, ax):
    runs = groups[name]["runs"]
    color = groups[name]["color"]

    datas = []

    for run in runs:
        data = requestJson(command + "/costs/" + run, "")
        costs = []
        accs = []
        for d in data:
            cost = d["cost"]
            acc = d["acc_mcts_moves"]
            if cost is not None and acc is not None:
                costs.append(cost)
                accs.append(acc)
        datas.append((costs, accs))

    if len(datas) > 1:
        for cost, acc in datas:
            plt.plot(cost, acc, color=whitenColor(color, 0.3), linewidth=1)

        meanX, meanY = meanInterpolatedLine(datas)
        plt.plot(meanY, meanX, label=name, color=color+(1,), linewidth=2)

        #annote_max(meanX, meanY, ax)

        mostLeft = meanY[0] * 0.99
        mostRight = meanY[-1] * 1.01
    elif len(datas) == 1:
        mostLeft = datas[0][0][0] * 0.99
        mostRight = datas[0][0][0] * 1.01
        plt.plot(datas[0][0], datas[0][1], color=whitenColor(color, 0.9), linewidth=1, label=name)

        #annote_max(datas[0][0], datas[0][1], ax)

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

    for groupName in groups:
        mL, mR = plotGroup(groupName, fig, ax)

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

    plt.ylim([yLow, yHigh])

    plt.legend(loc="lower right", fancybox=True)

    plt.title(plot_name)

    plt.ylabel("MCTS accuracy %")
    plt.xlabel("Estimated cost in hours")

    plt.savefig(img_output, bbox_inches="tight", format="eps", dpi=300)

    plt.show()