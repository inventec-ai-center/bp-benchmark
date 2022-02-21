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

## Processing

This preprocessing pipeline allows the user to prepare the datasets to their use in the training pipeline later on. Starting from the original-raw datasets downloaded in `bp_benchmark/datasets/raw/` , the pipeline performs the preprocessing steps of formating and segmentation, cleaning of distorted signals, feature extraction an data splitting for model validation.

The `process.py` script is in charge of performing the complete preparation of the datasets. To prepare a specific dataset, run the script with its correspoding configuration file at `--config_file` parameter. All config files can be found in `/bp-benchmark/code/process/core/config`. The following example shows the command for  the preparation of sensors dataset: 

```  bash
# Go to /bp-benchmark/code/process
cd /bp-benchmark/code/process
python process.py --config_file core/config/sensors_process.yaml
```

The processed datasets are saved in the directories indicated in the config files of each processing steps at `/bp-benchmark/code/process/core/config`. By default, the datasets are saved under the directory `/bp-benchmark/datasets` in the following structure:
- `./raw`: gathers the directories with the original data of each dataset (BCG, PPGBP, sensors & UCI).
- `./segmented`: stores the segmented datasets.
- `./preprocessed`: keeps the cleaned datasets (signals and features).
- `./splitted`: stores the splitted data ready for training and validation.

#### Modules

Besides, the code has been modularized to be able to perform each of the data preparation steps independently. There are different three modules:
- Segmenting module: reads, aligns and segments the raw data according to the config file passed as `--config_file` parameter.
    - There are one script for each dataset with the name `read_<data-name>.py`.
    - All config files of segmenting module are in `./core/config/segmentation`
```  bash
# In /bp-benchmark/code/process directory
python read_sensors.py --config_file core/config/segmentation/sensors_read.yaml
```
- Preprocessing module: cleans the segmented signals and extracts the PPG's features.
    - `cleaning.py` is the main script of the module. For PPG-BP dataset, please use `cleaningPPGBP.py`.
    - Its config files are located in `./core/config/preprocessing`.
```  bash
# In /bp-benchmark/code/process directory
python cleaning.py --config_file core/config/preprocessing/sensors_clean.yaml
```
- Splitting module: splits the data according to the validation strategy chosen in the config file of `--config_file` parameter.
     - `data_splitting.py` is the main script of the module.
    - All config files are in `./core/config/splitting`. 
```  bash
# In /bp-benchmark/code/process directory
python data_splitting.py --config_file core/config/preprocessing/sensors_clean.yaml
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
