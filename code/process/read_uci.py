import numpy as np
import pandas as pd
import mat73

import os
import argparse
from omegaconf import OmegaConf

from tqdm import tqdm

from core.lib.preprocessing import align_pair

def main(args):

    ## Create the dirs for the ouput data if do not exist
    os.makedirs(os.path.dirname(args.save_name), exist_ok=True)

    path_data_UCI = args.data
    fs=args.fs
    win_sec = args.win_sec
    win_sam = args.win_sam
    win_wait = args.win_wait*fs
    if win_sec != None:
        win_size = win_sec*fs
    elif win_sam != None:
        win_size = win_sam
    else:
        raise ValueError('Give value to either win_sec or win_sam')   

    if win_wait < win_size:
        win_wait = win_size
        
    print('Parameters -- win_size: {}, win_wait: {}, fs: {}'.format(win_size,args.win_wait,fs))
    
    dfs = [] 
    parts = ['Part_'+str(i) for i in range(1,5)]
    
    print('Reading the data, aligning and segmenting with win_size: {} and win_wait {} ...'.format(win_size,win_wait))
    
    dfs = [] 
    parts = ['Part_'+str(i) for i in range(1,5)]
    for part in parts:
        print(part,'/4')
        tmp_data = mat73.loadmat(path_data_UCI+part+'.mat')[part]
        dic = {}
        dic['patient'] = []
        dic['trial'] = []
        dic['signal'] = []
        dic['abp_signal'] = []


        for i in tqdm(range(len(tmp_data))):
            rec = tmp_data[i]
            a_abp, a_rppg, shift = align_pair(rec[1], rec[0], int(rec.shape[1]/fs), fs)

            n_seg = int(a_abp.shape[0] / win_wait)

            for j in range(n_seg):
                dic['patient'].append(part+'_'+str(i))
                dic['trial'].append(part+'_'+str(i)+'_'+str(j))
                dic['signal'].append(a_rppg[j*win_wait: j*win_wait + win_size])
                dic['abp_signal'].append(a_abp[j*win_wait: j*win_wait + win_size])

        dfs.append(pd.DataFrame(dic))
    
    df = pd.concat(dfs).reset_index(drop=True)
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

    


