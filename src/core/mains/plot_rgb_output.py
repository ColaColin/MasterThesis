# this does not really help much with anything...

from impls.polices.pytorch.policy import ResCNN, unpackTorchNetwork
import pytorch_model_summary as pms
from core.training.NetworkApi import NetworkApi
from utils.prints import logMsg, setLoggingEnabled
import setproctitle
from impls.games.connect4.connect4 import Connect4GameState

import matplotlib.pyplot as plt

import torch

import numpy as np

def createGame(moves):
    game = Connect4GameState(7,6,4)
    for move in moves:
        game = game.playMove(move-1)
    return game

def normArray(ar):
    minimum = np.min(ar)
    ar -= minimum
    maximum = np.max(ar)
    ar /= maximum
    return ar

def forwardGameState(net, game):
    tinput = torch.zeros((1, ) + game.getDataShape())
    npInput = tinput.numpy()
    game.encodeIntoTensor(npInput, 0, False)
    
    with torch.no_grad():
        processed = net(tinput)

    print(processed)

    print(processed[0][0].numpy().tolist())

    resultNumpy = processed[0].numpy().squeeze(0)

    print(resultNumpy.shape)

    # resultNumpy = np.moveaxis(resultNumpy, 0, 2)

    # red = normArray(resultNumpy[:,:,0]).reshape((6,7,1))
    # green = normArray(resultNumpy[:,:,1]).reshape((6,7,1))
    # blue = normArray(resultNumpy[:,:,2]).reshape((6,7,1))

    # renormed = np.concatenate((red, green, blue), axis=2)

    # # print(tinput.numpy(), "=>", processed[3].numpy())

    # # print(resultNumpy)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(resultNumpy[0,:,:])
    # ax2.imshow(resultNumpy[1,:,:])
    # ax3.imshow(resultNumpy[2,:,:])

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()

    #plt.imshow(renormed)

    plt.show()

if __name__ == "__main__":

    networkID = "f7b37c26-9568-4789-a996-5d662000f91f"
 
    setproctitle.setproctitle("x0_plot_rgb")
    setLoggingEnabled(True)

    networks = NetworkApi(noRun=True)

    net = ResCNN(6, 7, 1, 3, 128, 32, 3, 7, 3, 1, mode="sq", outputExtra="bothhead")
    # net = ResCNN(6, 7, 1, 3, 64, 16, 1, 7, 3, 8, mode="sq", outputExtra="bothhead")
    print(pms.summary(net, torch.zeros((1, 1, 6, 7))))

    networkData = networks.downloadNetwork(networkID)

    uuid, modelDict, ignoreConfig = unpackTorchNetwork(networkData)

    # net.load_state_dict(modelDict)

    game = createGame([4,4,4,4,4,4,5,3,6,7])
    # game = createGame([])
    # game = createGame([4,4,3,4,2])
    print(str(game))

    forwardGameState(net, game)
