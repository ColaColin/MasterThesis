import matplotlib.pyplot as plt


# fig = plt.figure(figsize=(8, 5))
# axL = plt.subplot(1,1,1)

# axM = axL.twinx()
# axM.set_ylabel("Momentum")

# axL.set_ylabel("Learning rate")

# axL.plot([0, 0.35, 0.7, 1], [0.02, 0.2, 0.02, 0.003], label="Learning rate", color=(1,0,0))
# axM.plot([0, 0.35, 0.7, 1], [0.95, 0.85, 0.95, 0.95], label="Momentum", color=(0,1,0))

# axL.legend(loc="lower left", fancybox=True)
# axM.legend(loc="center right", fancybox=True)

# plt.title("Cyclic learning rate and momentum")

# plt.savefig("/ImbaKeks/git/MasterThesis/Write/images/cyclic.eps", bbox_inches="tight", format="eps", dpi=300)

# plt.show()


fig = plt.figure(figsize=(8, 5))
axL = plt.subplot(1,1,1)

axL.set_ylabel("Inversion Hyperparameter")
axL.set_xlabel("Generation")

axL.plot([1] + list(range(3, 11)), [0.5, 0.149, 0.084, 0.071, 0.052, 0.047, 0.038, 0.029, 0.027], label="Inversion hyperparameter", color=(1,0,0))

axL.legend(loc="lower left", fancybox=True)

plt.title("Reduction of the inversion hyperparameter")

plt.savefig("/ImbaKeks/git/MasterThesis/Write/images/inversion_reduction.eps", bbox_inches="tight", format="eps", dpi=300)

plt.show()