import pandas as pd
import numpy as np
import sys
sys.path.append('../code/process/')

import os
import argparse
from omegaconf import OmegaConf

import joblib
from joblib import Parallel, delayed

from core.lib.preprocessing import normalize_data, mean_filter_normalize, my_find_peaks, identify_out_pk_vly, rm_baseline_wander

from core.lib.features_extraction import compute_sp_dp, extract_cycle_check, extract_feat_cycle, extract_feat_original, vpg_points, apg_points

from pyampd.ampd import find_peaks
import scipy.signal

from data_splitting import main as split

def _extract_c(sig, fs, pk_th=0.6, remove_start_end=True):
    """ Wrapper for extract_cycle_check function. """
    try:
        cs, pks_norm, flag1, flag2, pks, vlys = extract_cycle_check(sig, fs, pk_th, remove_start_end)
    except:
        cs, pks_norm, flag1, flag2, pks, vlys = [], [], True, True, [], []
    return cs, pks_norm, flag1, flag2, pks, vlys

def limit_cycle_len(cs, limit):
    return [c[:np.min([len(c), limit])] for c in cs]

def pad_to_max(cs):
    new_cs = []
    lens =  [len(c) for c in cs]
    max_len = np.max(lens)
    
    for l, c in zip(lens, cs):
        new_cs.append(np.pad(c, (0,max_len-l) , mode='constant', constant_values=np.nan))
    
    return np.stack(new_cs)

def mean_cycle(cs, mu, sigma, th):
    
    if len(cs) > 1:
        l = []
        for i in range(len(cs[0])):
            samples = cs[:,i]
            l.append(np.nanmean(samples[(samples >= mu[i]-th*sigma[i]) & (samples <= mu[i]+th*sigma[i])]))
        return np.array(l)
    else:
        return np.nanmean(cs,0)

def simple_z(cycle):
    
    peak = np.argmax(cycle[:int(len(cycle)*0.45)])
    vpg = np.gradient(cycle)
    
    end = int((len(vpg)-peak)*0.5)+peak
    y = np.argmin(vpg[peak:end])+peak
    z = np.argmax(vpg[y:end+1])+y
    
    return z

def to_mat(data, save_path):
    from scipy.io import loadmat, savemat 
    import re

    
    for i, d in enumerate(data):
        #--- reset index
        d = d.reset_index(drop=True)
        #--- to dictionary
        ddd = {}
        for col in d.columns:
            ddd[col] = np.vstack(d[col].values)
        
        #--- to .mat    
        savepath = save_path+f"/signal_mabp_fold_{i}.mat"
        savemat(savepath, ddd)
        
        #--- check if contents are maintained
        # reload the saved .mat
        ddd2 = loadmat(savepath)  
        # pop useless info
        ddd2.pop('__header__')
        ddd2.pop('__version__')
        ddd2.pop('__globals__')
        # convert to dataframe
        df = pd.DataFrame()
        for k, v in ddd2.items():
            v = list(np.squeeze(v))
            # deal with trailing whitespace
            if isinstance(v[0], str):
                v = [re.sub(r"\s+$","",ele) for ele in v]
            # convert string nan to float64
            v = [np.nan if ele=='nan' else ele for ele in v]
            # df[k] = list(np.squeeze(v))
            df[k] = v
        COLNAME = d.columns



if __name__=="__main__":

    #---------- Read config file ----------#
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path for the config file", required=True) 
    args_m = parser.parse_args()

    ## Read config file
    if os.path.exists(args_m.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args_m.config_file))
    args = OmegaConf.load(args_m.config_file)
    
    LIMIT_CYCLE = args.limit_cycle # 149
    ADDED = args.added # 15
    THRESHOLD = args.threshold # 1.25
    
    
    df = pd.read_pickle(args.path.data) ## Read dataset
    original_columns = df.columns ## Save original columns
    
    print('---- Identify cycle ----')
    cycle_st = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(_extract_c)(sig, fs=args.fs, remove_start_end=False) for sig in df.abp_signal)

    for i, label in enumerate(['cs','pks_norm','abp_f1','abp_f2','abp_pks','abp_vlys']):
        df[label] = [val[i] for val in cycle_st]
        
        
    print(f'f1: {df.abp_f1.sum()}, f2: {df.abp_f2.sum()}, nocy: {(df.cs.map(len)==0).sum()}')
    print('f1: {:.2f}, f2: {:.2f}, nocy: {:.2f}'.format(df.abp_f1.sum()/df.shape[0] * 100, df.abp_f2.sum()/df.shape[0] * 100, (df.cs.map(len)==0).sum()/df.shape[0] * 100))
        
        
    print('---- Limit and cycle ----')
    df['cs'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(limit_cycle_len)(cs, LIMIT_CYCLE) for cs in df.cs)
    df['pad_cs'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(pad_to_max)(cs) for cs in df.cs)
    
    print('---- Compute mu and sigma ----')
    df['mu'] = df['pad_cs'].map(lambda c:  np.nanmean(c,0))
    df['sigma'] = df['pad_cs'].map(lambda c:  np.nanstd(c,0))
    
    print('---- Compute mean ABP ----')
    df['mean_abp'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(mean_cycle)(row.pad_cs, row.mu, row.sigma, THRESHOLD) for i, row in df.iterrows())
    
    print('---- Compute points (peak, notch, end) ----')
    df['sys_peak'] = df['mean_abp'].map(lambda cycle: np.argmax(cycle[:int(len(cycle)*0.45)]))
    df['dia_notch'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(simple_z)(cy) for cy in df.mean_abp)
    df['SP_mod'] = df['mean_abp'].map(max)
    df['DP_mod'] = df['mean_abp'].map(min)

    df['class_limits'] = [[row.sys_peak, row.dia_notch, len(row.mean_abp), len(row.mean_abp)+ADDED] for i, row in df.iterrows()]
    
    print('---- Extend cycle ----')
    lim = LIMIT_CYCLE + ADDED
    df['abp_signal'] = df.mean_abp.map(lambda c: np.pad(c,(0,lim-len(c)), mode='wrap'))

    print('---- Save data ----')
    data_df=df[['patient','trial','SP','DP','signal','abp_signal','class_limits']]
    data_df.to_pickle(args.path.save_name)
    
    # Splitting
    if 'splitting' in args.keys():
        print('---- Splitting data ----')
        split(args.splitting)
        
        data_df = joblib.load(args.splitting.path.split_df_path)
        to_mat(data_df, args.splitting.path.mat_path)
        
        

   
        
    
    
    
    
    