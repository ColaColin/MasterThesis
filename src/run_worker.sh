#! /bin/bash

# meant to work as root in nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 docker container
# download the script, run it with a few parameters and the worker should startup automatically after installing dependencies
# Start it in two steps like this:

# 1.) apt install -y wget; wget https://raw.githubusercontent.com/ColaColin/MasterThesis/master/src/run_worker.sh ; chmod +x run_worker.sh
# 2.) ./run_worker.sh <number of workers> <command server> <secret> <runid> <workername>

# apt install -y wget; wget https://raw.githubusercontent.com/ColaColin/MasterThesis/master/src/run_worker.sh ; chmod +x run_worker.sh; ./run_worker.sh <number of workers> https://x0.cclausen.eu <secret> <runid> <workername>

#e.g. ./run_worker.sh 5 https://x0.cclausen.eu <secret> <runid> <workername>

apt update
apt install -y git
apt install -y gcc
apt install -y g++

wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

chmod +x Anaconda3-2019.10-Linux-x86_64.sh

./Anaconda3-2019.10-Linux-x86_64.sh -b -p /root/anaconda3

export PATH="/root/anaconda3/bin:$PATH"
eval "$('/root/anaconda3/bin/conda' 'shell.bash' 'hook')"

conda init

conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch

git clone https://github.com/ColaColin/MasterThesis.git
cd MasterThesis
SHA=$(wget -qO- $2/sha/$4)
git checkout $SHA
cd src

pip install -r req_workers.txt

chmod +x build.sh

./build.sh

./build.sh testworker

for ((i=1; i <= $1; i++))
do
    python -m core.mains.distributed --command $2 --secret $3 --run $4 --worker $5 --windex $i &
done

