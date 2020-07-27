import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 5))
axL = plt.subplot(1,2,1)
axR = plt.subplot(1,2,2)

p90Data = {
    "2": 9,
    "3": 17,
    "4": 48,
    "5": 49,
    "6": 104,
    "7": 120,
    "8": 173,
    "9": 209,
    "10": 347,
    "11": 264,
    "12": 309,
    "13": 355,
    "14": 411,
    "15": 434,
    "16": 424,
    "17": 482,
    "18": 417,
    "19": 443,
    "20": 413,
    "21": 424,
    "22": 361,
    "23": 404,
    "24": 365,
    "25": 356,
    "26": 334,
    "27": 311,
    "28": 313,
    "29": 287,
    "30": 257,
    "31": 255,
    "32": 221,
    "33": 233,
    "34": 219,
    "35": 199,
    "36": 178,
    "37": 137,
    "38": 98,
    "39": 17,
    "40": 3
}

p50Data = {
    "0": 1,
    "1": 2,
    "2": 9,
    "3": 56,
    "4": 217,
    "5": 345,
    "6": 769,
    "7": 674,
    "8": 992,
    "9": 788,
    "10": 833,
    "11": 663,
    "12": 620,
    "13": 585,
    "14": 478,
    "15": 426,
    "16": 403,
    "17": 351,
    "18": 268,
    "19": 229,
    "20": 200,
    "21": 179,
    "22": 157,
    "23": 147,
    "24": 108,
    "25": 95,
    "26": 69,
    "27": 72,
    "28": 50,
    "29": 40,
    "30": 46,
    "31": 34,
    "32": 25,
    "33": 16,
    "34": 15,
    "35": 15,
    "36": 14,
    "37": 4,
    "39": 5
}

def toHData(inDict):
    result = []
    for k in dict.keys(inDict):
        kn = int(k)
        kv = inDict[k]
        for _ in range(kv):
            result.append(kn)
    return result

axL.hist(toHData(p50Data), density=True, bins=len(dict.keys(p50Data)))
axR.hist(toHData(p90Data), density=True, bins=len(dict.keys(p90Data)))

axL.set_ylim([0, 0.105])
axR.set_ylim([0, 0.105])

axL.set_title("harder dataset: 50% random moves")
axR.set_title("easier dataset: 90% random moves")

fig.text(0.5, 0.04, 'Example in turn', ha='center')
fig.text(0.04, 0.5, 'Fraction of positions at that turn in dataset', va='center', rotation='vertical')

plt.savefig("/ImbaKeks/git/MasterThesis/Write/images/dataset_hist.eps", bbox_inches="tight", format="eps", dpi=300)


plt.show()

