#! /bin/bash

# Run with parameter "cpu" to install cpuonly pytorch!
# Othewise pytorch-cuda is installed

wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

chmod +x Anaconda3-2019.10-Linux-x86_64.sh

./Anaconda3-2019.10-Linux-x86_64.sh

source ~/.bashrc

if [[ $# -eq 0 ]] ; then
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
elif [ $1 == "cpu" ]; then
    conda install pytorch torchvision cpuonly -c pytorch
fi

pip install -r requirements.txt

chmod +x build.sh

./build.sh

./build.sh test

