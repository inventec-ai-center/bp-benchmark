# BP-Algorithm

## Setup environment
``` bash
#-- Create folder
mkdir bp_benchmark
cd bp_benchmark

#-- Clone project
git clone git@gitlab.com:inventecaicenter/bp-algorithm.git
cd bp-algorithm
# Go to the branch
git checkout exp/sensors

#-- Download data
mkdir -p datasets/splits
cd datasets/splits 
# sftp to NAS, download share_mat/* to datasets/splits/

#-- Build docker image
cd ../..  # back to /bp-algorithm
docker build -t bpimage .
docker run --gpus=all --shm-size=65g --name=bp_bm -p 9180-9185:9180-9185 -it -v ~/bp_benchmark/bp-algorithm/:/sensorsbp bpimage bash

#-- Quick test the environment
python train.py --config_file core/config/ml/lgb/lgb_bcg_SP.yaml
```
