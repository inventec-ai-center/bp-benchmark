import numpy as np
import pandas as pd

import os
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

    ## Create the dirs for the ouput data if do not exist
    os.makedirs(os.path.dirname(args.save_name), exist_ok=True)

    path_data_BCG = args.data
    
    print('Reading the data...')
    bed_df = pd.read_csv(path_data_BCG+'bcg_pat_info.csv')


    fppg = []
    freBAP = []
    for i in tqdm(range(1,41)):
        pat_pred_df = pd.read_csv(path_data_BCG+'pats/pat'+str(i)+'pred.csv')
        fppg.append(pat_pred_df['PPG'].values)
        freBAP.append(pat_pred_df['reBAP'].values)

    bed_df['PPG'] = fppg
    bed_df['reBAP'] = freBAP

    for n_sig in ['PPG','reBAP']:
        bed_df[n_sig] = bed_df[n_sig].map(lambda sig: sig[:-1000])
    
    if args.resampling != -1:
        print('Resampling the data...')
        for name in ['PPG','reBAP']:
            bed_df[name] = bed_df[name].map(lambda sig: my_resample(sig, old_fs=args.fs, new_fs=args.resampling))
        fs = args.resampling
    else:
        fs = args.fs

    print('Cut 3 sec beginning and end...')
    sec = 3
    win = sec*fs
    for name in ['PPG','reBAP']:
        bed_df[name] = bed_df[name].map(lambda sig: sig[win:-win])


    print('Scale back reBAP')
    bed_df['reBAP']=bed_df.reBAP*100
    
    print('Aligning the data...')
    ### Align the signal
    for i, row in bed_df.iterrows():
        
        a_abp, a_rppg, shift = align_pair(row.reBAP, row.PPG, int(len(row.reBAP)/fs), fs)

        bed_df.at[i,'reBAP'] = a_abp
        bed_df.at[i,'PPG'] = a_rppg


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


    print('Segmenting with win_size: {} and win_wait {} ...'.format(win_size,win_wait))

    dfs = []
    for i, row in bed_df.iterrows():
        #n_seg = int(row.reBAP.shape[0] / win_size)
        n_seg = int(row.reBAP.shape[0] / win_wait)

        for j in range(n_seg):
            vals = list(row[['ID', 'Gender', 'Age', 'Height_cm', 'Weight_kg', 'HeartCondition',
           'Comments']].values)
            for sig_n in ['PPG','reBAP']:
                vals=vals + [row[sig_n][j*win_wait: j*win_wait + win_size]]  
                
            vals=vals + [row['ID']+'_'+str(j)]
            dfs.append(vals)

    df=pd.DataFrame(dfs, columns=list(bed_df.columns)+['trial'])

    #columns order
    columns = ['ID','trial', 'Gender', 'Age', 'Height_cm', 'Weight_kg', 'HeartCondition',
       'Comments', 'PPG','reBAP']

    df = df[columns]

    df = df.rename(columns={'ID':'patient','PPG':'signal', 'reBAP':'abp_signal'})

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
    
    


