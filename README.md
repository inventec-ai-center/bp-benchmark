# BP-Algorithm

## Setup environment
``` bash
#-- Create folder
mkdir bp_benchmark
cd bp_benchmark

#-- Download data 
mkdir -p ./datasets/splits
cd ./datasets/splits 
# sftp to NAS, download share_mat/* to datasets/splits/

#-- Clone project
cd ../.. # back to /bp-benchmark
git clone git@gitlab.com:inventecaicenter/bp-algorithm.git
cd bp-algorithm
# Go to the branch
git checkout exp/sensors

#-- Build docker image
docker build -t bpimage .
docker run --gpus=all --shm-size=65g --name=bp_bm -p 9180-9185:9180-9185 -it -v ~/bp_benchmark/bp-algorithm/:/sensorsbp -v ~/bp_benchmark/datasets/:/sensorsbp/datasets bpimage bash

#-- Quick test the environment
cd code/train
python train.py --config_file core/config/ml/lgb/lgb_bcg_SP.yaml
```

## Training

- Training Feat2Lab models: 
	- Input the desired config file with --config_file argument
	- All the config files of Feat2Lab approach are in `./core/config/ml/` 

```
# Go to /sensorsbp/code/train
cd /sensorsbp/code/train
python train.py --config_file core/config/ml/lgb/lgb_bcg_SP.yaml
```

- Training Sig2Lab models
	- All the config files of Feat2Lab approach are in `./core/config/dl/resnet/` 

```
# Go to /sensorsbp/code/train
cd /sensorsbp/code/train
python train.py --config_file core/config/dl/resnet/resnet_bcg.yaml
```

- Training Sig2Sig models
	- All the config files of Feat2Lab approach are in `./core/config/dl/unet/` 

```
# Go to /sensorsbp/code/train
cd /sensorsbp/code/train
python train.py --config_file core/config/dl/resnet/unet_bcg.yaml
```

## Tuning hyper-parameters with Hydra Optuna
- Tuning Sig2Lab or Sig2Sig models
	- Edit the config file's path in `tune.py`
```
#============== In tune.py ==============#
# Edit the config file's path in following line:
@hydra.main(config_path='./core/config/tune/dl', config_name="resnet_bcg_tune")

#======= In command line interface ======#
# Go to /sensorsbp/code/train
cd /sensorsbp/code/train
python tune.py -m
```
