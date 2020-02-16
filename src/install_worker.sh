#! /bin/bash

apt update
apt install gcc

wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

chmod +x Anaconda3-2019.10-Linux-x86_64.sh

./Anaconda3-2019.10-Linux-x86_64.sh

source ~/.bashrc

conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch

pip install -r req_workers.txt

chmod +x build.sh

./build.sh

./build.sh testworker