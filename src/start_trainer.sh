#! /bin/bash

# starting the trainer is down a machine that the dependencies were installed on by hand
# but the checkout should be set to the commit of the run configuation
# thus start the training process with this script

# apt install -y wget; wget https://raw.githubusercontent.com/ColaColin/MasterThesis/master/src/start_trainer.sh ; chmod +x start_trainer.sh
# ./start_trainer.sh https://x0.cclausen.eu <secret> <runid>

# do it in some place for temporary files, this creates a new checkout!

git clone https://github.com/ColaColin/MasterThesis.git
cd MasterThesis
SHA=$(wget -qO- $1/sha/$3)
git checkout $SHA
cd src

chmod +x build.sh

./build.sh

./build.sh testworker

python -m core.mains.distributed --command $1 --secret $2 --run $3 --training