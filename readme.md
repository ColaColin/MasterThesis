This is the code used in my Master's Thesis "Investigation of possible improvements to increase the efficiency of the AlphaZero algorithm."
The image below shows the basic architecture of the framework. All code is in the folder src.

![Architecture overview](x0_framework_overview.png)

# Setup

## Setup the central server

To reproduce my results you will need to setup the central server that manages the experimental runs, called command. The machine used should have at least 8, better 16GB of RAM and preferably a quad core CPU to handle the data transfers and statistics analysis done on the server. HDD size should be a few GB per training run.

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

# Reproducability

To be able to reproduce experiments as I developed the code I kept the exact SHA of the commits used for certain experiments. 

# Specific experiments

It follows a description of the various kinds of experiments and how to do them using the code.

## 
