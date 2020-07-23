This is the code used in my Master's Thesis "Investigation of possible improvements to increase the efficiency of the AlphaZero algorithm."
The image below shows the basic architecture of the framework. All code is in the folder src.

![Architecture overview](x0_framework_overview.png)

# Setup

## Setup the central server

To reproduce my results you will need to setup the central server that manages the experimental runs, called command. The machine used should have at least 8, better 16GB of RAM and preferably a quad core CPU to handle the data transfers and statistics analysis done on the server. HDD size should be a few GB per training run. There is a backup.sh script to do backups from the remote system to a local system, though it would need to be modified for your server name. On request I can share the backup of the server with the exact experiments data I created, but it is about 200GB big, so I cannot easily share it on the internet.

1. The server is meant to run on ubuntu 18.04, other OS are not tested
2. Install postgres (10.x), python 3.7 anaconda, nginx, and for some of the experiments, node.js.
3. Checkout this repository on the server and install the dependencies outlined in src/req_command.txt
4. Setup nginx and configure it as outlined in the example file in configs/nginx. The python based server will run on port 8042, nginx will add SSL and reverse proxy. You could skip nginx, if you don't want to bother with nginx and let's encrypt SSL.
5. Setup a database in postgres using the file in src/setup.sql
6. The server needs a configuration file called server.json, there is in an example in configs/server.json. Change the config values to fit your system, databaseuse, etc. dataPath is a directory where binary data will be stored (stored networks and self-play training examples). Secret is a simple password used in the APIs.
7. You can then start the command server like so: python -m core.mains.command --config server.json

## Setup self-play workers

For self-play workers using vast.ai there is a script in src/run_worker2.sh which can be used as a startup script on vast.ai. It contains more comments on how exactly that works. If you want to start workers elsewhere, you can read the script and repurpose it, all necessary setup of dependencies, in the context of the pytorch 1.5 docker container, is done in the script. Workers should have about 12GB RAM per GPU. You also need 3-5 CPU cores per GPU, depending on the GPU. Less CPUs yield suboptimal GPU loads, though that certainly depends a lot on the exact configuration. 

## Setup the trainer

The training happens on a single machine, the process must not be restarted during a run, so it is better to use a machine under your full control.
Checkout this repository and install python 3.7 anaconda and the dependencies listed in src/req_workers.txt as a minimum.
Then start the trainer for a specific run: python -m core.mains.distributed --command server-path --secret 42 --run run-id --training

It should start to download positions for the run and push new networks to the server. The training code keeps all positions in memory as it trains, so you need a decent amount of RAM on the training machine.

You need to balance the number of GPUs producing training example with how fast your training machine is. If you have too many workers, training will fall behind.

# Reproducibility

To be able to reproduce experiments as I developed the code I kept the exact SHA of the commits used for experiments. Runs are configured in the command server
with the SHA and a config file. All run configs can be found in configs/run. Below is a table of which run IDs were involved for which experiments described by my Thesis.  To reproduce an experiment, pick the config with the filename, use the SHA listed at the top of the file in command. The worker script automatically uses the SHA configured in the command server, for the trainer you need to switch by hand.

| Experiment      | IDs |
| ----------- | ----------- |
| Baseline      | 12c31999-e8a9-4a52-b017-796e64b05f8a, 45d2087b-04f9-49ca-b2d9-5c8736da86b5, 59e19f40-c4f0-46e9-97f8-5a2b423ef7fc, bdf69def-4476-43fc-934d-115a7d895d6e, 1edc288e-df3e-47c1-b9ce-52ab0045404a           |
| Extended baseline | 7d675f5e-0926-43f9-b508-a55b06a42b2c, 5c3f34d0-deae-4aa4-a6c1-be6ecb9d4e86, b9336ccf-69e1-4ad4-8a5a-246e734d7a81, e2f7655f-94f4-4e58-9397-a3b8d11ef5d8, 658f11a2-a862-418e-a3b5-32d145d3dbdf |
| Deduplication with weight 0.2 | 505b834a-345a-4a1b-a67a-fa405b27d6e4 |
| Deduplication with weight 0.5 | 43a39db5-5eec-43d9-9e50-a945248a64e8 |
| Deduplication with weight 0.8 | 8b1900b0-d133-4435-baf5-6c35934ff94c |
| Cyclic learning rate | 7d434f56-e7c0-4945-af3b-3abdb30f4fca |
| Slow training window | 32bb62a4-5541-4c0c-af1d-e84c09dfdccc |
| Playout Caps | 0538a5d8-0706-4b90-b601-c0adbfd69cc6 |
| Predict the opponent's reply | fd514ad3-35db-44e9-8768-76c5822dc09e | 
| Squeeze and Excite Resnet | f64aae6e-f094-47b5-aa0c-1201b324e939 |
| hyperopt1 hyperparameters | aa4782ae-c162-4443-a290-41d7bb625d17, 3d09cdce-4e69-4811-bda9-ad2985228230, fe764034-ba1f-457b-8329-b5904bb8f66c, 3eca837f-4b4d-439e-b6e7-09b35edf3d5d, 55efa347-a847-4241-847e-7497d2497713   |
| hyperopt2 hyperparameters |  1edc288e-df3e-47c1-b9ce-52ab0045404a, bdf69def-4476-43fc-934d-115a7d895d6e, 59e19f40-c4f0-46e9-97f8-5a2b423ef7fc, 45d2087b-04f9-49ca-b2d9-5c8736da86b5, 12c31999-e8a9-4a52-b017-796e64b05f8a |
| prevWork hyperparameters | 65388799-c526-4870-b371-fb47e35e56af, 583eae5c-94e8-4bdb-ac48-71bc418c5a51, bba7494a-c5f9-42bb-90ff-b93a91b5e74b, 5350bdc8-8c4b-422b-8bfc-0435d2b6d45d, 9733ab7c-7ebc-49eb-87db-1f03e0929d10 |
| player evolution cpuct, fpu, drawValue | 325d9f51-97d2-48ab-8999-25f2583979ba |
| player evolution kldgain | 1a4c1c39-a812-4f82-9da4-17bf237baeb7 |
| novelty search: points for novel wins | 395ba2f7-4e40-4cbd-ae57-19087d344e25 |
| novelty search: points for novel positions | f8830ef7-0e14-4e0d-ae29-87378daf5b5f, d2e2917f-4ca3-4c13-89d0-ebdf2ca152e6, 037fa6cc-4975-459d-9a84-98ce9eb1342d |
| cached MCTS is a more efficient implementation | 1d182bb0-5b26-49fb-b2a9-4417322f76e5, d91dbba7-4363-4779-8a86-7a127977d9e4, e1500cbb-45ae-4e1a-a55a-8015fa414afd |
| Explore by retrying different moves after losses | e6135ef6-e360-47d7-b9bb-bfe91f3a341b |
| Exploration by MCTS, cpuct = 15 | e5eb7ac2-3123-46bd-a79a-5026814a859c |

Most of the charts were made by the script src/core/table_printer. It still contains the configurations used in commented-out form. It needs a command server to pull data from.

# Experiments specifics

It follows a description of the various kinds of experiments and how to do them using the code. Some extra steps are needed, depending on the experiment.
The configurations work by using the name property to name classes, all other keys in the nodes are then used as constructor parameters. This works in a recursive fashion, by using $ as a prefix to the name of nodes as parameter to other nodes. Look at some of the examples as described above.

## Baseline runs

No special steps need to be taken to do baseline runs. The difference between the baseline and the extended baseline is some switches in the config file.
For examples, look at the config files of the runs, as listed above.

### Deduplication

Deduplication is controlled by the flags deduplicate and deduplcationWeight in the trainer configuration

### Cyclic learning rate

The cycle of learning rate and momenum are controlled by objects with name "OneCycleSchedule", see the example configurations. They are referenced by the PytorchPolicy configuration via the keys momentumDecider and lrDecider

### Improved training window

The improved training window is an alternative implementation of the WindowSizeManager, called SlowWindowSizeManager.
It is provided to the trainer as windowManager.

### Playout Caps

Playout caps are implemented by having two nodes that represent a MCTSPolicyIterator, one for the full search, one for the smaller, cheaper search.
They are then provided to the LinearSelfPlayWorker, relevant properties are capPolicyP, capPolicyIterator, capMoveDecider

### Sequeeze and Excite Networks

Controlled by a switch on the PytorchPolicy: networkMode, set it to sq

### Prediction of move replies.

Controlled by a switch on the PytorchPolicy: replyWeight, 0 disables it, higher values give a weight to the loss in the training function.

## Evolutionary hyperparameter optimization

The central server tracks the players and provides APIs for this. The server runs some additional code for this, which is controlled by the serverLeague node in those configurations. The workers use the LeagueSelfPlayerWorker implementation.

### Novelty search

Novelty search requires an additional server process, written in javascript for node.js, which accepts reports of novel game positions. It keeps tracks of all md5 hashes of all positions ever encountered, which can consume a few GB of memory on full runs. Start it on the command server via "node src/node_server/novelty-service.js"

## Playing games as trees

These configurations require an additional management server that manages the high-throughput MCTS evaluation service. Due to performance issues with python this is written in javascript, using node.js. The management server needs to be started by hand for each experiment via "node src/node_server/eval-service.js" on the command server.

## Using network features of a small network to regularize a big network

This is controlled by a few flags and requires no additional processes. On the PytorchPolicy you set useWinFeatures or useMoveFeatures. They can be set to values -1,0,1,2,3 to use either no features or present/future features. -1 disables, 0 uses just present features. Using values above 0 uses the combination of the present features and all future features up to the given turn. Alternatively a list can be provided to specify only specific future turns to be used.

To produce features the LinearSelfPlayWorker needs to be provided an additional ResCNN node configured for the exact network and a featureNetworkID, which is the UUID of the network to be used, which should exist on the command server.
