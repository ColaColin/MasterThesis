#! /bin/bash

# since setting up new command servers is not a common task:
# this is not tested well and still needs you to setup postgres (and the databases in it) by hand before

apt update
apt install gcc
apt install libpq-dev

wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

chmod +x Anaconda3-2019.10-Linux-x86_64.sh

./Anaconda3-2019.10-Linux-x86_64.sh

source ~/.bashrc

conda install -y pytorch torchvision cpuonly -c pytorch

pip install -r req_command.txt

chmod +x build.sh

./build.sh

./build.sh test

