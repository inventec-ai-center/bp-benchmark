A Benchmark for Machine-Learning based Non-Invasive Blood Pressure Estimation using Photoplethysmogram <br />
<sub>Sergio González, Wan-Ting Hsieh, and Trista Pei-Chun Chen</sub>
------------

Blood Pressure (BP) is an important cardiovascular health indicator. BP is usually monitored noninvasively with a cuff-based device, which can be bulky and inconvenient. Thus, continuous and portable BP monitoring devices, such as those based on a photoplethysmography (PPG) waveform, are desirable. In particular, Machine Learning (ML) based BP estimation approaches have gained considerable attention as they have the potential to estimate intermittent or continuous BP with only a single PPG measurement. Over the last few years, many ML-based BP estimation approaches have been proposed with no agreement on their modeling methodology. To ease the model comparison, we designed a benchmark with four open datasets with shared preprocessing, the right validation strategy avoiding information shift and leak, and standard evaluation metrics. We also adapted Mean Absolute Scaled Error (MASE) to improve the interpretability of model evaluation, especially across different BP datasets. The proposed benchmark comes with open datasets and codes. We showcase its effectiveness by comparing 11 ML-based approaches of three different categories.

# Installation

## Setup environment
``` bash
#-- Create folder
mkdir bp_benchmark
cd bp_benchmark

#-- Download data 
mkdir -p ./datasets/splitted
cd ./datasets/splitted
# sftp to NAS, download share_mat/* to datasets/splits/

#-- Download trained models 
cd ../..
mkdir ./models
cd ./models
# sftp to NAS, download BP-data-model/* to models/

#-- Clone project
cd ../.. # back to /bp-benchmark
git clone git@gitlab.com:inventecaicenter/bp-algorithm.git
cd bp-algorithm
# Go to the branch
git checkout exp/sensors

#-- Build docker image
docker build -t bpimage .
docker run --gpus=all --shm-size=65g --name=bp_bm -p 9180-9185:9180-9185 -it -v ~/bp_benchmark/bp-algorithm/:/bp_benchmark -v ~/bp_benchmark/datasets/:/bp_benchmark/datasets -v ~/bp_benchmark/models/:/bp_benchmark/models bpimage bash

#-- Quick test the environment
cd code/train
python train.py --config_file core/config/ml/lgb/lgb_bcg_SP.yaml
```

## Datasets

Before running the training pipeline, the user could directly download the preprocessed datasets (**recommended**), or download raw datasets and process them.

#### Preprocessed datasets

The preprocessed datasets might be found in the [<repo-name>'s website](). They should be located under the directory `/bp_benchmark/datasets/splitted/` to proceed with the training stage.

```bash
mkdir /bp_benchmark/datasets/splitted/
cd /bp_benchmark/datasets/splitted/

# Download the dataset under /bp_benchmark/datasets/splitted/
wget -O data.zip <link-to-figshare>
# Unzip the data
unzip data.zip 
    
# OPTIONAL: remove unnecessary files
rm -r data.zip 
```

#### Raw datasets

The raw datasets should be located under the directory `/bp_benchmark/datasets/raw/` to proceed with the processing stage.

- sensors dataset might be found in [Zenodo's website](https://zenodo.org/record/4598938):
``` bash
mkdir /bp_benchmark/datasets/raw/sensors
cd /bp_benchmark/datasets/raw/sensors

# Download sensors dataset under datasets/raw/sensors
wget https://zenodo.org/record/4598938/files/ABP_PPG.zip
# Unzip the data
unzip ABP_PPG.zip
# move files to desired path
mv ABP_PPG/* . 

# OPTIONAL: remove unnecessary files
rm -r ABP_PPG ABP_PPG.zip 'completed (copy).mat' completed.mat 
```
- UCI dataset might be downloaded from [UCI repository](https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation#).
```bash
mkdir /bp_benchmark/datasets/raw/UCI
cd /bp_benchmark/datasets/raw/UCI

# Download UCI dataset under datasets/raw/UCI
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00340/data.zip
# Unzip the data
unzip data.zip 

```
- BCG dataset has to be downloaded manually from [IEEE*DataPort*'s site](https://ieee-dataport.org/open-access/bed-based-ballistocardiography-dataset) with an IEEE account. 
```bash
mkdir /bp_benchmark/datasets/raw/BCG
cd /bp_benchmark/datasets/raw/BCG

# > Manually download the data from IEEEDataPort and 
# > Locate 'Data Files.zip' under /bp_benchmark/datasets/raw/BCG

# Unzip the data
unzip 'Data Files.zip' 
# move files to desired path
mv 'Data Files'/* .

# OPTIONAL: remove unnecessary files
rm -r 'Data Files'
```
Before proceeding with the processing stage of BCG dataset, the Matlab's script `code/process/core/bcg_mat2csv.m` must be run with the raw data path as parameter. 
```bash
cd /bp_benchmark/code/process/core/bcg_mat2csv.m
matlab -r "bcg_mat2csv('../../../datasets/raw/BCG')"
```

- PPGBP dataset might be found in [Figshare's site](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299).
```bash
mkdir /bp_benchmark/datasets/raw/PPGBP
cd /bp_benchmark/datasets/raw/PPGBP

# Download PPGBP dataset under datasets/raw/PPGBP
wget -O data.zip https://figshare.com/ndownloader/files/9441097
# Unzip the data
unzip data.zip 
# move files to desired path
mv 'Data File'/* .

# OPTIONAL: remove unnecessary files
rm -r 'Data File' data.zip 
```

# How To Use   
 
## Processing

This preprocessing pipeline allows the user to prepare the datasets to their use in the training pipeline later on. Starting from the original-raw datasets downloaded in `bp_benchmark/datasets/raw/` , the pipeline performs the preprocessing steps of formating and segmentation, cleaning of distorted signals, feature extraction and data splitting for model validation.

<!-- The `process.py` script is in charge of performing the complete preparation of the datasets. To prepare a specific dataset, run the script with its correspoding configuration file at `--config_file` parameter. All config files can be found in `/bp_benchmark/code/process/core/config`. The following example shows the command for  the preparation of sensors dataset:  -->
`process.py --config_file [CONFIG PATH]`
* It performs the complete preparation of the datasets. (segmenting -> cleaning -> splitting)
* [CONFIG PATH]: provide the path of a config file, all config files can be found in `/bp_benchmark/code/process/core/config`
```  bash
# Go to /bp_benchmark/code/process
cd /bp_benchmark/code/process
python process.py --config_file core/config/sensors_process.yaml
```

The processed datasets are saved in the directories indicated in the config files of each processing steps at `/bp_benchmark/code/process/core/config`. By default, the datasets are saved under the directory `/bp_benchmark/datasets` in the following structure:
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
# In /bp_benchmark/code/process directory
python read_sensors.py --config_file core/config/segmentation/sensors_read.yaml
```
- Preprocessing module: cleans the segmented signals and extracts the PPG's features.
    - `cleaning.py` is the main script of the module. For PPG-BP dataset, please use `cleaningPPGBP.py`.
    - Its config files are located in `./core/config/preprocessing`.
```  bash
# In /bp_benchmark/code/process directory
python cleaning.py --config_file core/config/preprocessing/sensors_clean.yaml
```
- Splitting module: splits the data according to the validation strategy chosen in the config file of `--config_file` parameter.
     - `data_splitting.py` is the main script of the module.
    - All config files are in `./core/config/splitting`. 
```  bash
# In /bp_benchmark/code/process directory
python data_splitting.py --config_file core/config/preprocessing/sensors_clean.yaml
```


## Training
`train.py --config_file [CONFIG PATH]`
* It trains the model with the parameters set in the input config file.
* [CONFIG PATH]: provide the path of a config file, all config files can be found in `/bp_benchmark/code/train/core/config`
#### Training Feat2Lab models: 

- All the config files of Feat2Lab approach are in `./core/config/ml/` 

```
# Go to /bp_benchmark/code/train
cd /bp_benchmark/code/train
python train.py --config_file core/config/ml/lgb/lgb_bcg_SP.yaml
```

#### Training Sig2Lab models

- All the config files of Feat2Lab approach are in `./core/config/dl/resnet/` 

```
# Go to /bp_benchmark/code/train
cd /bp_benchmark/code/train
python train.py --config_file core/config/dl/resnet/resnet_bcg.yaml
```

#### Training Sig2Sig models

- All the config files of Feat2Lab approach are in `./core/config/dl/unet/` 

```
# Go to /bp_benchmark/code/train
cd /bp_benchmark/code/train
python train.py --config_file core/config/dl/unet/unet_bcg.yaml
```

The models are save in the path indicated in the config_file (`${path.model_directory}`), by default it will be in `/bp_benchmark/code/train/model-${exp.model_type}`.

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
# Go to /bp_benchmark/code/train
cd /bp_benchmark/code/train
python tune_ml.py -m
```

#### Tuning Sig2Lab or Sig2Sig models

- Edit the config file's path in `tune.py`
```
#============== In tune.py ==============#
# Edit the config file's path in following line:
@hydra.main(config_path='./core/config/tune/dl', config_name="resnet_bcg_tune")

#======= In command line interface ======#
# Go to /bp_benchmark/code/train
cd /bp_benchmark/code/train
python tune.py -m
```

## Testing
`test.py --config_file [CONFIG PATH]`
* It tests on test set with the trained models specified in the input config file.
* [CONFIG PATH]: provide the path of a config file, all config files can be found in `/bp_benchmark/code/train/core/config/test`

#### Testing Sig2Lab or Sig2Sig models

- All the config files of Feat2Lab approach are in `./core/config/test/dl/` 

```
# Go to /bp_benchmark/code/train
cd /bp_benchmark/code/train
python test.py --config_file core/config/test/dl/ts_rsnt_bcg.yaml
```

## Checking results in MLflow
``` bash
# Setup mlflow server in background

# Step 1: open a new session with tmux for mlflow
tmux new -s mlflow

# Step 2: in the new session setup mlflow server
cd /bp_benchmark/code/train
mlflow ui -h 0.0.0.0 -p 9181 --backend-store-uri ./mlruns/
# leave the session with ^B+D
 
# Step 3: forward the port 9181
ssh -N -f -L localhost:9181:localhost:9181 username@working_server -p [port to working_server] -i [ssh key to working_server]

# Step 4: now you can browse mlflow server in your browser
# open a new tab in your browser and type http://localhost:9181/
```

# Copyright Information

This project is under the terms of [the MIT license](https://opensource.org/licenses/mit-license.php). It is only for research or education purposes, and not freely available for commercial use or redistribution. This intellectual property belongs to the Inventec Corporation. Licensing is possible if you want to use the code for commercial use. For scientific use, please reference this repository together with the relevant publication.

```
@article{BPbenchmark2023,
    author   = {González, Sergio and Hsieh, Wan-Ting and Chen, Trista Pei-Chun},
    title    = {A Benchmark for Machine-Learning based Non-Invasive Blood Pressure Estimation using Photoplethysmogram},
    journal  = {Scientific Data)},
    year     = {2023},
    numpages = {19}
}
```
