import numpy as np
import pandas as pd
from copy import deepcopy

import os
import argparse
from omegaconf import OmegaConf
import scipy.io

from tqdm import tqdm

from core.lib.preprocessing import align_pair

def main(args):

    ## Create the dirs for the ouput data if do not exist
    os.makedirs(os.path.dirname(args.save_name), exist_ok=True)

    print('Reading data...')
    df = pd.read_csv(args.rec2subid)
    path = args.data
    data_list = []
    for file in tqdm(os.listdir(path)):
        if file in df.file_name.values:
            
            #print(os.path.join(path, file))
            #print(file)
            patient = int(df[df.file_name==file].subject_id.iloc[0])
            gender = df[df.file_name==file].gender.iloc[0]
            age = df[df.file_name==file].age.iloc[0]
            #print(patient)
            record = file[:-4]
            #print(record)
            loaded_file = scipy.io.loadmat(os.path.join(path, file))
            assert loaded_file['signal_processing'][0].shape == (2,)
            
            for segment in [0,1]:
                dic_vals_0 = {}
                dic_vals_0['patient'] = patient
                dic_vals_0['trial'] = record+'_'+str(segment)
                dic_vals_0['gender'] = gender
                dic_vals_0['age'] = age

                ## first element info
                dic_vals_0['signal'] = loaded_file['signal_processing'][0][segment][0][0]
                dic_vals_0['abp_signal'] = loaded_file['signal_processing'][0][segment][0][1]

                data_list.append(pd.Series(dic_vals_0))

    data = pd.DataFrame(data_list)
    data = data.sort_values(['patient','trial']).reset_index(drop=True)



    fs=args.fs
    win_sec = args.win_sec
    win_sam = args.win_sam
    if win_sec != None:
        win_size = win_sec*fs
    elif win_sam != None:
        win_size = win_sam
    else:
        raise ValueError('Give value to either win_sec or win_sam')   
    
    df_ori=data
    
    df = deepcopy(df_ori)
    
    print('Parameters -- win_size: {}, fs: {}'.format(win_size,fs))
    
    print('Aligning and segmenting with win_size: {}...'.format(win_size)) 
    
    for i in tqdm(range(df.shape[0])):
        abp = df.iloc[i].abp_signal
        ppg = df.iloc[i].signal
        a_abp, a_rppg, shift = align_pair(abp, ppg, int(len(abp)/fs), fs)
        
        init_idx = (len(a_abp) - win_size)//2
        
        df['signal'].iloc[i] = a_rppg[init_idx: init_idx+win_size]
        df['abp_signal'].iloc[i] = a_abp[init_idx: init_idx+win_size]
        
    print('Removing duplicates...')
    num_prev = df.shape[0]
    df['ppg_str'] = df.signal.map(lambda p: p.tobytes())
    df['abp_str'] = df.abp_signal.map(lambda p: p.tobytes())
    df = df.drop_duplicates(subset='ppg_str', keep='first').reset_index(drop=True)
    df = df.drop_duplicates(subset='abp_str', keep='first').reset_index(drop=True)
    df.drop(columns=['ppg_str','abp_str'], inplace=True)
    
    print('Removed duplicates: {}'.format(num_prev-df.shape[0]))
    
    print('Saving data...')
    df.to_pickle(args.save_name)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path for the config file", required=True)
    args_m = parser.parse_args()
    
    if os.path.exists(args_m.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args_m.config_file))

    config = OmegaConf.load(args_m.config_file)
    main(config)



    
    
    
    