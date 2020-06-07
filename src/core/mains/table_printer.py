# generate diagrams and stuff from data on the server
# just hack in the right values and run the script...

from utils.req import requestJson
import setproctitle

from utils.prints import logMsg, setLoggingEnabled

import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

plot_name = "foobar"

groups = {
    "test": {
        "color": (1,0,0),
        "runs": [
            "0538a5d8-0706-4b90-b601-c0adbfd69cc6",
            "8b1900b0-d133-4435-baf5-6c35934ff94c"
        ]
    }
}

command = "http://127.0.0.1:8042"
img_output = "/ImbaKeks/git/MasterThesis/Write/images/"

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
    
    if idx == 0 or idx == len(yData):
        return None

    xAfter = xData[idx]
    xBefore = xData[idx-1]

    yAfter = yData[idx]
    yBefore = yData[idx-1]

    assert yPoint <= yAfter and yPoint >= yBefore

    yRange = yAfter - yBefore
    linScale = (yPoint - yBefore) / yRange

    return xBefore + (xAfter - xBefore) * linScale

def plotGroup(name, fig, ax):
    runs = groups[name]["runs"]
    color = groups[name]["color"]

    datas = []

    for run in runs:
        data = requestJson(command + "/costs/" + run, "")
        costs = list(map(lambda x: x["cost"], data))
        accs = list(map(lambda x: x["acc_mcts_moves"], data))
        datas.append((costs, accs))

    if len(datas) > 1:
        for cost, acc in datas:
            plt.plot(cost, acc, color=color + (0.62, ), linewidth=1)

        meanX, meanY = meanInterpolatedLine(datas)
        plt.plot(meanY, meanX, label=name, color=color+(1,), linewidth=2)
        mostLeft = meanY[0] * 0.99
        mostRight = meanY[-1] * 1.01
    elif len(datas) == 1:
        mostLeft = datas[0][0][0] * 0.99
        mostRight = datas[0][0][0] * 1.01
        plt.plot(datas[0][0], datas[0][1], color=color + (0.9, ), linewidth=1, label=name)

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

    plt.xlim([mostLeft, mostRight])

    plt.legend(loc="lower right", fancybox=True)

    plt.title(plot_name)

    plt.grid()
    plt.ylabel("MCTS accuracy %")
    plt.xlabel("Estimated cost in hours")
    plt.show()