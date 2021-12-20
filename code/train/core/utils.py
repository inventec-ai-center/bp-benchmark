import os 
from pathlib import Path
import argparse
import torch
import torch.nn
import numpy as np
import mlflow as mf
from shutil import rmtree
from pyampd.ampd import find_peaks
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils.safety import try_mlflow_log
    
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def set_device(gpu_id):
    print(gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        print("Using GPU: ", torch.cuda.current_device())
    else:
        n_threads = torch.get_num_threads()
        n_threads = min(n_threads, 8)
        torch.set_num_threads(n_threads)
        print("Using {} CPU Core".format(n_threads))

def get_nested_fold_idx(kfold):
    for fold_test_idx in range(kfold):
        fold_val_idx = (fold_test_idx+1)%kfold
        fold_train_idx = [fold for fold in range(kfold) if fold not in [fold_test_idx, fold_val_idx]]
        yield fold_train_idx, [fold_val_idx], [fold_test_idx]

def get_ckpt(r):
    ckpts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "restored_model_checkpoint")]
    return r.info.artifact_uri, ckpts

#%% Global Normalization
def global_norm(x, signal_type): 
    if signal_type == "sbp": (x_min, x_max) = (60, 200)   # mmHg
    elif signal_type == "dbp": (x_min, x_max) = (30, 110)   # mmHg
    elif signal_type == "ptt": (x_min, x_max) = (100, 900)  # 100ms - 900ms
    else: return None

    # Return values normalized between 0 and 1
    return (x - x_min) / (x_max - x_min)
    
def global_denorm(x, signal_type):
    if signal_type == "sbp": (x_min, x_max) = (60, 200)   # mmHg
    elif signal_type == "dbp": (x_min, x_max) = (30, 110)   # mmHg
    elif signal_type == "ptt": (x_min, x_max) = (100, 900)  # 100ms - 900ms
    else: return None

    # Return values normalized between 0 and 1
    return x * (x_max-x_min) + x_min

def glob_demm(x, config, type='sbp'): 
    # sensors global max, min
    if type=='sbp':
        x_min, x_max = config.param_loader.sbp_min, config.param_loader.sbp_max
    elif type=='dbp':
        x_min, x_max = config.param_loader.dbp_min, config.param_loader.dbp_max
    elif type=='ppg':
        x_min, x_max = config.param_loader.ppg_min, config.param_loader.ppg_max
    return x * (x_max-x_min) + x_min

def glob_mm(x, config, type='sbp'): 
    # sensors global max, min
    if type=='sbp':
        x_min, x_max = config.param_loader.sbp_min, config.param_loader.sbp_max
    elif type=='dbp':
        x_min, x_max = config.param_loader.dbp_min, config.param_loader.dbp_max
    elif type=='ppg':
        x_min, x_max = config.param_loader.ppg_min, config.param_loader.ppg_max
    return (x - x_min) / (x_max - x_min)

def glob_dez(x, config, type='sbp'): 
    # sensors global max, min
    if type=='sbp':
        x_mean, x_std = config.param_loader.sbp_mean, config.param_loader.sbp_std
    elif type=='dbp':
        x_mean, x_std = config.param_loader.dbp_mean, config.param_loader.dbp_std
    elif type=='ppg':
        x_mean, x_std = config.param_loader.ppg_mean, config.param_loader.ppg_std
    return x * (x_std + 1e-6) + x_mean

def glob_z(x, config, type='sbp'): 
    # sensors global max, min
    if type=='sbp':
        x_mean, x_std = config.param_loader.sbp_mean, config.param_loader.sbp_std
    elif type=='dbp':
        x_mean, x_std = config.param_loader.dbp_mean, config.param_loader.dbp_std
    elif type=='ppg':
        x_mean, x_std = config.param_loader.ppg_mean, config.param_loader.ppg_std
    return (x - x_mean)/(x_std + 1e-6)

#%% Local normalization
def loc_mm(x,config, type='sbp'):
    return (x - x.min())/(x.max() - x.min() + 1e-6)

def loc_demm(x,config, type='sbp'):
    return x * (x.max() - x.min() + 1e-6) + x.min()

def loc_z(x,config, type='sbp'):
    return (x - x.mean())/(x.std() + 1e-6)

def loc_dez(x,config, type='sbp'):
    return x * (x.std() + 1e-6) + x.mean()

#%% Compute bps
def compute_sp_dp(sig, fs=125, pk_th=0.6):
    sig = sig.astype(np.float64)
    peaks = find_peaks(sig,fs)
    valleys = find_peaks(-sig,fs)
    
    sp, dp = -1 , -1
    flag1 = False
    flag2 = False
    
    ### Remove first or last if equal to 0 or len(sig)-1
    if peaks[0] == 0:
        peaks = peaks[1:]
    if valleys[0] == 0:
        valleys = valleys[1:]
    
    if peaks[-1] == len(sig)-1:
        peaks = peaks[:-1]
    if valleys[-1] == len(sig)-1:
        valleys = valleys[:-1]
    
    '''
    ### HERE WE SHOULD REMOVE THE FIRST AND LAST PEAK/VALLEY
    if peaks[0] < valleys[0]:
        peaks = peaks[1:]
    else:
        valleys = valleys[1:]
        
    if peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
    else:
        valleys = valleys[:-1]
    '''
    
    ### START AND END IN VALLEYS
    while len(peaks)!=0 and peaks[0] < valleys[0]:
        peaks = peaks[1:]
    
    while len(peaks)!=0 and peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
    
    ## Remove consecutive peaks with one considerably under the other
    new_peaks = []
    mean_vly_amp = np.mean(sig[valleys])
    if len(peaks)==1:
        new_peaks = peaks
    else:
        # define base case:

        for i in range(len(peaks)-1):
            if sig[peaks[i]]-mean_vly_amp > (sig[peaks[i+1]]-mean_vly_amp)*pk_th:
                new_peaks.append(peaks[i])
                break

        for j in range(i+1,len(peaks)):
            if sig[peaks[j]]-mean_vly_amp > (sig[new_peaks[-1]]-mean_vly_amp)*pk_th:
                new_peaks.append(peaks[j])
                
        if not np.array_equal(peaks,new_peaks):
            flag1 = True
            
        if len(valleys)-1 != len(new_peaks):
            flag2 = True
            
        if len(valleys)-1 == len(new_peaks):
            for i in range(len(valleys)-1):
                if not(valleys[i] < new_peaks[i] and new_peaks[i] < valleys[i+1]):
                    flag2 = True
        
        
    return np.median(sig[new_peaks]), np.median(sig[valleys]), flag1, flag2, new_peaks, valleys
    
def get_bp_pk_vly_mask(data):
    _,_,_,_,pks, vlys = compute_sp_dp(data, 125, pk_th=0.6)

    pk_mask = np.zeros_like(data)
    vly_mask = np.zeros_like(data)
    pk_mask[pks] = 1
    vly_mask[vlys] = 1
    
    return np.array(pk_mask, dtype=bool), np.array(vly_mask, dtype=bool)

#%% Compute statistics for normalization
def cal_statistics(config, all_df):
    import pandas as pd
    from omegaconf import OmegaConf,open_dict
    all_df = pd.concat(all_df)
    OmegaConf.set_struct(config, True)

    with open_dict(config):
        for x in ['sbp', 'dbp']:
            config.param_loader[f'{x}_mean'] = float(all_df[x].mean())
            config.param_loader[f'{x}_std'] = float(all_df[x].std())
            config.param_loader[f'{x}_min'] = float(all_df[x].min())
            config.param_loader[f'{x}_max'] = float(all_df[x].max())
        
        # ppg
        config.param_loader[f'ppg_mean'] = float(np.vstack(all_df['sfppg']).mean())
        config.param_loader[f'ppg_std'] = float(np.vstack(all_df['sfppg']).std())
        config.param_loader[f'ppg_min'] = float(np.vstack(all_df['sfppg']).min())
        config.param_loader[f'ppg_max'] = float(np.vstack(all_df['sfppg']).max())
    return config

#%% Compute metric
def cal_metric(err_dict, metric={}, mode='val'):
    for k, v in err_dict.items():
        metric[f'{k}_mae'] = np.mean(np.abs(v))
        metric[f'{k}_std'] = np.std(v)
        metric[f'{k}_me'] = np.mean(v)
    metric = {f'{mode}/{k}':round(v.item(),3) for k,v in metric.items()}
    return metric

#%% print/logging tools
def print_criterion(sbps, dbps):
    print("The percentage of SBP above 160: (0.10)", len(np.where(sbps>=160)[0])/len(sbps)) 
    print("The percentage of SBP above 140: (0.20)", len(np.where(sbps>=140)[0])/len(sbps)) 
    print("The percentage of SBP below 100: (0.10)", len(np.where(sbps<=100)[0])/len(sbps)) 
    print("The percentage of DBP above 100: (0.05)", len(np.where(dbps>=100)[0])/len(dbps)) 
    print("The percentage of DBP above 85: (0.20)", len(np.where(dbps>=85)[0])/len(dbps)) 
    print("The percentage of DBP below 70: (0.10)", len(np.where(dbps<=70)[0])/len(dbps)) 
    print("The percentage of DBP below 60: (0.05)", len(np.where(dbps<=60)[0])/len(dbps)) 

def get_cv_logits_metrics(fold_errors, loader, pred_sbp, pred_dbp, pred_abp, 
                            true_sbp, true_dbp, true_abp, 
                            sbp_naive, dbp_naive, mode="val"):

    fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
    fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
    fold_errors[f"{mode}_sbp_naive"].append([sbp_naive])
    fold_errors[f"{mode}_sbp_pred"].append([pred_sbp])
    fold_errors[f"{mode}_sbp_label"].append([true_sbp])
    fold_errors[f"{mode}_dbp_naive"].append([dbp_naive])
    fold_errors[f"{mode}_dbp_pred"].append([pred_dbp])
    fold_errors[f"{mode}_dbp_label"].append([true_dbp])
    fold_errors[f"{mode}_abp_true"].append([true_abp])
    fold_errors[f"{mode}_abp_pred"].append([pred_abp])

#%% mlflow
def init_mlflow(config):
    mf.set_tracking_uri(str(Path(config.path.mlflow_dir).absolute()))  # set up connection
    mf.set_experiment(config.exp.exp_name)          # set the experiment

def log_params_mlflow(config):
    mf.log_params(config.get("exp"))
    # mf.log_params(config.get("param_feature"))
    try_mlflow_log(mf.log_params, config.get("param_preprocess"))
    try_mlflow_log(mf.log_params, config.get("param_trainer"))
    try_mlflow_log(mf.log_params, config.get("param_early_stop"))
    mf.log_params(config.get("param_loader"))
    # mf.log_params(config.get("param_trainer"))
    # mf.log_params(config.get("param_early_stop"))
    if config.get("param_aug"):
        if config.param_aug.get("filter"):
            for k,v in dict(config.param_aug.filter).items():
                mf.log_params({k:v})
    # mf.log_params(config.get("param_aug"))
    mf.log_params(config.get("param_model"))

def log_config(config_path):
    # mf.log_artifact(os.path.join(os.getcwd(), 'core/config/unet_sensors_5s.yaml'))
    # mf.log_dict(config, "config.yaml")
    mf.log_artifact(config_path)

def log_hydra_mlflow(name):
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), f'{name}.log'))
    rmtree(os.path.join(os.getcwd()))