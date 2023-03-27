#%% 
import random
import numpy as np
import pandas as pd
import joblib

from omegaconf import OmegaConf 
import argparse
import os

from sklearn.model_selection import KFold
import warnings


def get_nested_fold_idx(kfold):
    for fold_test_idx in range(kfold):
        fold_val_idx = (fold_test_idx+1)%kfold
        fold_train_idx = [fold for fold in range(kfold) if fold not in [fold_test_idx, fold_val_idx]]
        yield fold_train_idx, [fold_val_idx], [fold_test_idx]


def data_splitting(config):
    # ------------------------------------------------------------------------------------------
    # cross-validation
    if config.param_split.type=='cv': 
        data = pd.read_pickle(config.path.processed_df_path)
        print()
        warnings.warn("Please refer to Figshare split datasets or Scientific Data paper for self-implementation. "+
            "This procedure keeps the samples from the same patient in the same fold. " + 
            "But it does not preserve the BP distributions between the folds.")
        print()
        subs = data.patient.unique()
        ksplitter = KFold(n_splits=config.param_split.fold, shuffle=True, random_state=config.param_split.random_state)
        all_split_df = []
        for train, test in ksplitter.split(subs):
            test_df = data[data.patient.isin(subs[test])].reset_index(drop=True)
            all_split_df.append(test_df)
        
    # ------------------------------------------------------------------------------------------
    # Hold-one-out
    elif config.param_split.type=='hoo':
        df = pd.read_pickle(config.path.processed_df_path)
        df["part"] = df.apply(lambda row: '_'.join(row.patient.split('_')[:2]), axis=1)
        
        # Separate tr and val/ts by partitions
        tr_df = df[df.part.isin(config.param_split.tr_part)]
        if config.param_split.val_part==config.param_split.ts_part: 
            # split 20% of data by records as validation 
            val_ts_df = df[df.part.isin(config.param_split.val_part)]

            # Split val and ts by records
            all_recs = list(val_ts_df.patient.unique())

            val_recs = random.sample(all_recs, int(0.2*len(all_recs)))

            val_df = val_ts_df[val_ts_df.patient.isin(val_recs)]
            exclude = val_ts_df.index.isin(val_df.index.to_list())
            ts_df = val_ts_df[~exclude]
        else: 
            val_df = df[df.part.isin(config.param_split.val_part)]
            ts_df = df[df.part.isin(config.param_split.ts_part)]

        ts_df = ts_df.drop(columns=['part'])
        val_df = val_df.drop(columns=['part'])
        tr_df = tr_df.drop(columns=['part'])

        all_split_df = [ts_df, val_df, tr_df]

    for i in range(len(all_split_df)): ### remove 'label_class' column
        if 'label_class' in all_split_df[i].columns:
            all_split_df[i].drop(columns=['label_class'], inplace=True)
        if 'agg_ohe' in all_split_df[i].columns:
            all_split_df[i].drop(columns=['agg_ohe'], inplace=True)

    joblib.dump(all_split_df, config.path.split_df_path)

def main(config):
    np.random.seed(config.param_split.random_state)
    random.seed(config.param_split.random_state)

    ## Create the dirs for the ouput data if do not exist
    os.makedirs(os.path.dirname(config.path.split_df_path), exist_ok=True)

    data_splitting(config)

    # load result and get naive
    all_split_df = joblib.load(config.path.split_df_path)
    
    all_sp_MAE = []
    all_dp_MAE = []
    all_sp_ME = []
    all_dp_ME = []
    for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(len(all_split_df))):
        if (config.param_split.type=='hoo') and foldIdx!=0:  break
        print(folds_train, folds_val, folds_test)
        train_df = pd.concat(np.array(all_split_df, dtype=object)[folds_train])
        val_df = pd.concat(np.array(all_split_df, dtype=object)[folds_val])
        test_df = pd.concat(np.array(all_split_df, dtype=object)[folds_test])

        train_sp = train_df["SP"].mean()
        train_dp = train_df["DP"].mean()
        all_sp_MAE.append((train_sp-test_df["SP"]).abs())
        all_dp_MAE.append((train_dp-test_df["DP"]).abs())
        all_sp_ME.append(train_sp-test_df["SP"])
        all_dp_ME.append(train_dp-test_df["DP"])
    all_sp_MAE = np.hstack(all_sp_MAE)
    all_dp_MAE = np.hstack(all_dp_MAE)
    all_sp_ME = np.hstack(all_sp_ME)
    all_dp_ME = np.hstack(all_dp_ME)
    sp_mae = np.mean(all_sp_MAE)
    dp_mae = np.mean(all_dp_MAE)
    print("SBP: MAE {:.4f} | ME {:.4f} | STD {:.4f}".format(sp_mae, all_sp_ME.mean(), all_sp_ME.std()))
    print("DBP: MAE {:.4f} | ME {:.4f} | STD {:.4f}".format(dp_mae, all_dp_ME.mean(), all_dp_ME.std()))
    cnt = 0
    for d in all_split_df:
        cnt+=d.shape[0]
    print('total amount', cnt)

#%%
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path for the splitting config file", required=True) 
    args_m = parser.parse_args()

    ## Read config file
    if os.path.exists(args_m.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args_m.config_file))

    # split the data with given config
    config = OmegaConf.load(args_m.config_file)

    main(config)

    

    
    