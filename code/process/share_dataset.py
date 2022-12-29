"""
This scripts transform the files generated with the processing pipeline to .mat/.csv, 
which was the format used to share the datasets.
"""


#%% Save to HDF5
import pandas as pd
import numpy as np
import joblib
import h5py

DATA_PATH = '../../datasets/splitted/'
FEA_PATH = '../../datasets/splitted/'

#%% Save to .mat
import pandas as pd
import numpy as np
import joblib
from scipy.io import loadmat, savemat 
import re

                
#datasets = ['PPGBP', 'sensors', 'BCG', 'UCI']
datasets = ['UCI']
for dataset in datasets:
    dataset0 = dataset if dataset!='UCI' else 'uci2'
    dataset0 = dataset0.lower()
    data = joblib.load(f'{DATA_PATH}/{dataset}.pkl')
    for i, d in enumerate(data):
        print('---------------------------------------')
        print(dataset, i)
        #--- reset index
        d = d.reset_index(drop=True)
        #--- to dictionary
        ddd = {}
        for col in d.columns:
            ddd[col] = np.vstack(d[col].values)
        
        #--- to .mat    
        savepath = f"{DATA_PATH}/{dataset0}_dataset/signal_fold_{i}.mat"
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
            
        COLNAME = d.columns #if dataset!='ppgbp' else ['patient', 'signal', 'SP', 'DP', 'Diabetes', 'trial', 'Sex(M/F)']
        for col in COLNAME:
            print(col, df[col].equals(d[col]))
        
        
#%% save features to csv
import joblib      
import pandas as pd
from pandas.testing import assert_frame_equal
                
#datasets = ['PPGBP', 'sensors', 'BCG', 'UCI']
datasets = ['UCI']
for dataset in datasets:
    dataset0 = dataset if dataset!='UCI' else 'uci2'
    dataset0 = dataset0.lower()
    data = joblib.load(f'{DATA_PATH}/{dataset}_feats.pkl')
    for i, d in enumerate(data):
        print('---------------------------------------')
        print(dataset, i)
        #--- reset index
        test = d.reset_index(drop=True)
        savepath = f"{DATA_PATH}/{dataset0}_dataset/feat_fold_{i}.csv"
        #--- to csv
        test.to_csv(savepath, float_format='%.20f', index=False)
        # #--- check if contents are maintained
        # df = pd.read_csv(savepath,float_precision='high')
        # assert_frame_equal(df, test, check_exact=True)





