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

This preprocessing pipeline allows the user to prepare the datasets to their use in the training pipeline later on. Starting from the original-raw datasets downloaded in `bp_benchmark/datasets/raw/` , the pipeline performs the preprocessing steps of formating and segmentation, cleaning of distorted signals, feature extraction and data splitting for model validation.

<!-- The `process.py` script is in charge of performing the complete preparation of the datasets. To prepare a specific dataset, run the script with its correspoding configuration file at `--config_file` parameter. All config files can be found in `/bp-benchmark/code/process/core/config`. The following example shows the command for  the preparation of sensors dataset:  -->
`process.py --config_file [CONFIG PATH]`
* It performs the complete preparation of the datasets. (segmenting -> cleaning -> splitting)
* [CONFIG PATH]: provide the path of a config file, all config files can be found in `/bp-benchmark/code/process/core/config`
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

#### Processing's Modules

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
`train.py --config_file [CONFIG PATH]`
* It trains the model with the parameters set in the input config file.
* [CONFIG PATH]: provide the path of a config file, all config files can be found in `/bp-benchmark/code/train/core/config`
#### Training Feat2Lab models: 

- All the config files of Feat2Lab approach are in `./core/config/ml/` 

```
# Go to /sensorsbp/code/train
cd /sensorsbp/code/train
python train.py --config_file core/config/ml/lgb/lgb_bcg_SP.yaml
```

#### Training Sig2Lab models

- All the config files of Feat2Lab approach are in `./core/config/dl/resnet/` 

```
# Go to /sensorsbp/code/train
cd /sensorsbp/code/train
python train.py --config_file core/config/dl/resnet/resnet_bcg.yaml
```

#### Training Sig2Sig models

- All the config files of Feat2Lab approach are in `./core/config/dl/unet/` 

```
# Go to /sensorsbp/code/train
cd /sensorsbp/code/train
python train.py --config_file core/config/dl/resnet/unet_bcg.yaml
```

The models are save in the path indicated in the config_file (`${path.model_directory}`), by default it will be in `/sensorsbp/code/train/model-${exp.model_type}`.

## Tuning hyper-parameters with Hydra Optuna
`tune.py -m`
- The input of tune.py is the config_file which is assigned in this script. People can edit the config path in the script.
- The parameter search space is defined in the config_file with Hydra optuna tool.
- `-m` is the flag to do tune the parameters for multi run, without giving `-m`, it won't tune the parameters in search space, but only run with the paramters set in config_file (`${param_model}`) once.

#### Tuning Feat2Lab models
- Edit the config file's path in `tune_ml.py`
```
#============== In tune.py ==============#
# Edit the config file's path in following line:
@hydra.main(config_path='./core/config/tune/ml', config_name="lgb_sensors_SP")

#======= In command line interface ======#
# Go to /sensorsbp/code/train
cd /sensorsbp/code/train
python tune_ml.py -m
```

#### Tuning Sig2Lab or Sig2Sig models

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

## Checking results in MLflow
``` bash
# Setup mlflow server in background
# Step 1: install tmux (or screen) to run multiple sessions
apt-get update
apt-get install tmux

# Step 2: open a new session for mlflow
tmux new -s mlflow

# Step 3: in the new session setup mlflow server
cd /sensorsbp/code/train
mlflow ui -h 0.0.0.0 -p 9181 --backend-store-uri ./mlruns/
# leave the session with ^B+D
 
# Step 4: forward the port 9181
ssh -N -f -L localhost:9181:localhost:9181 username@working_server -p [port to working_server] -i [ssh key to working_server]

# Step 5: now you can browse mlflow server in your browser
# open a new tab in your browser and type http://localhost:9181/
```
