import numpy as np
import pandas as pd

import os
from os import listdir
from os.path import isfile, join

import argparse
from omegaconf import OmegaConf

from tqdm import tqdm

from core.lib.preprocessing import align_pair
from scipy.signal import resample

def my_resample(sig, old_fs, new_fs):
    ori_len = len(sig)
    re_size = int(np.round(ori_len/old_fs * new_fs))
    return resample(sig, re_size)

def main(args):
    path_data_PPGBP = args.data 

    df_info=pd.read_excel(path_data_PPGBP+'PPG-BP-dataset.xlsx')
    df_info = df_info.drop(columns='Num.')
    df_info = df_info.rename(columns={'subject_ID':'patient','Systolic Blood Pressure(mmHg)':'SP','Diastolic Blood Pressure(mmHg)':'DP'})


    print('Reading the data...')

    path_subjects = path_data_PPGBP+'0_subject/'
    df = {'patient':[],'trial':[],'signal':[]}
    for f in listdir(path_subjects):
        if isfile(join(path_subjects, f)):
            file = f
            pat_id = file.split('_')[0]
            trial = file.split('.')[0]
            ppg = pd.read_csv(join(path_subjects, f),sep='\t',header=None).values.flatten()[:-1]
            
            df['patient'].append(int(pat_id))
            df['trial'].append(trial)
            df['signal'].append(ppg)
    df = pd.DataFrame(df)
    df=pd.merge(df_info,df,on='patient') 

    #Ordering the columns
    labels = ['patient','trial','SP','DP']
    columns = list(df.columns)
    for l in labels:
        columns.remove(l)  
    df = df[['patient','trial','SP','DP']+columns]


    ## Resampling
    if args.resampling != -1:
        print('Resampling the data...')
        df['signal'] = df['signal'].map(lambda sig: my_resample(sig, old_fs=args.fs, new_fs=args.resampling))
    else:
        fs = args.fs
        

    print('Saving the data...')
    df.to_pickle(args.save_name)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path for the config file") 
    args_m = parser.parse_args()
    
    if os.path.exists(args_m.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args_m.config_file))
        
    config = OmegaConf.load(args_m.config_file)
    
    main(config)

    


