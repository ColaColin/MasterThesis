#! /bin/bash

# meant to work as root in pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime docker container

# Use this as the startup script on vast.ai:

# apt install -y wget; wget https://raw.githubusercontent.com/ColaColin/MasterThesis/master/src/run_worker2.sh ; chmod +x run_worker2.sh; ./run_worker2.sh https://x0.cclausen.eu <secret> <runid>

NET_DIR=/tmp/x0_networks
CHECK_FILE=installed.flag

if [[ -f "$CHECK_FILE" ]]
then
  echo "Not the first start on this machine, skipped installation"
  rm $NET_DIR -rf

else
  echo "Seems this is the first start here, installing x0 $SHA!"
  touch $CHECK_FILE

  apt update
  apt install -y git
  apt install -y gcc
  apt install -y g++

  git clone https://github.com/ColaColin/MasterThesis.git
  cd MasterThesis
  SHA=$(wget -qO- $1/sha/$3)
  git checkout $SHA
  cd src

  pip install -r req_workers.txt

  chmod +x build.sh

  ./build.sh

  ./build.sh testworker
fi

MAX_PER_GPU=4

if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q '1080 Ti'; then
  MAX_PER_GPU=5
fi

if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 2070 SUPER'; then
  MAX_PER_GPU=5
fi

if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q '2080'; then
  MAX_PER_GPU=5
fi

if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 2080 SUPER'; then
  MAX_PER_GPU=6
fi

if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q '2080 Ti'; then
  MAX_PER_GPU=6
fi

GPUC=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
MAX_BY_GPU=$((MAX_PER_GPU * GPUC))

MAX_BY_CPU=$(nproc)

WORKERS=$((MAX_BY_CPU > MAX_BY_GPU ? MAX_BY_GPU : MAX_BY_CPU))

nvidia-smi -l &>> /root/gpu_load.log &

for ((i=1; i <= $WORKERS; i++))
do
    echo python -u -m core.mains.distributed --command $1 --secret $2 --run $3 --worker $VAST_CONTAINERLABEL --windex $i '&>>' /root/worker_$i.log '&'
    python -u -m core.mains.distributed --command $1 --secret $2 --run $3 --worker $VAST_CONTAINERLABEL --windex $i &>> /root/worker_$i.log &
    sleep 1
done


