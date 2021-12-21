#%% 
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection import iterative_train_test_split
from collections import Counter
from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd
import joblib

def label2class(sbp, dbp):
    ret = np.zeros(2)
    # sbp to class
    if sbp<100:         ret[0] = 0
    elif 100<=sbp<140:  ret[0] = 1
    elif 140<=sbp<160:  ret[0] = 2
    else:               ret[0] = 3

    # dbp to class
    if dbp<60:          ret[1] = 0
    elif 60<=dbp<80:    ret[1] = 1
    elif 80<=dbp<100:   ret[1] = 2
    else:               ret[1] = 3

    return ret.astype(int)

def ohe(counter_result):
    ret = np.zeros(16)
    # encode (sbp_class, dbp_class) to another class, totally 16  classes (4 from sbp * 4 from dbp)
    ohe_dict = {str(np.array([i, j])): 4*i+j for i in range(4) for j in range(4)}
    for k, v in counter_result.items():
        ret[ohe_dict[k]] = v
    return ret.astype(int), ohe_dict

def get_nested_fold_idx(kfold):
    for fold_test_idx in range(kfold):
        fold_val_idx = (fold_test_idx+1)%kfold
        fold_train_idx = [fold for fold in range(kfold) if fold not in [fold_test_idx, fold_val_idx]]
        yield fold_train_idx, [fold_val_idx], [fold_test_idx]


def data_splitting(config):
    # ------------------------------------------------------------------------------------------
    # cross-validation
    if config.param_split.type=='cv': 
        data = pd.read_pickle(config.path.processed_df_path) #sensors_data2.pkl
        # data.columns = ['subject_id', 'trial', 'sbp', 'dbp', 'sfppg', 'abp']

        if config.path.demo_df_path != None:
            demo_df = pd.read_csv(config.path.demo_df_path)
            # drop useless column in demo_df
            demo_df = demo_df.drop(["Unnamed: 0"], axis=1)
            # make subject_id to string
            demo_df.subject_id = demo_df.subject_id.apply(lambda x: f'{x:06}')
            # merge demo_df to data
            data = data.merge(demo_df, on='patient', how='left')
            # data conversion
            data['sex'] = data.apply(lambda row: 0 if row["gender"]=='M' else 1, axis=1)
            min_age, max_age = data["age"].min(), data["age"].max()
            data['norm_age'] = data.apply(lambda row: (row["age"]-min_age)/(max_age-min_age), axis=1)

        if config.param_split.is_mix:
            data["label_class"] = data.apply(lambda row: label2class(row["SP"], row["DP"]), axis=1)
            data["agg_ohe"] = data.apply(lambda row: ohe({str(row["label_class"]):1})[0], axis=1)
            # Stratified spliting
            k_fold = IterativeStratification(n_splits=config.param_split.fold, order=1)
            all_split_df = []
            for train, test in k_fold.split(data, np.stack(data["agg_ohe"].values)):
                test_df = data.iloc[test]
                all_split_df.append(test_df)
        else:
            # convert label to class: (SBP, DBP) -> (SBP class, DBP class)    
            data["label_class"] = data.apply(lambda row: label2class(row["SP"], row["DP"]), axis=1)

            # count frequency of the same (SBP class, DBP class) pair in each subject
            all_subject = []
            all_agg_ohe = []
            for group in data.groupby(['patient']):
                sub_id, sub_df = group
                agg_ohe,ohe_dict = ohe(Counter(str(row) for row in sub_df['label_class'] ))
                all_agg_ohe.append(agg_ohe)
                all_subject.append(sub_id)

            # Stratified spliting
            new_sub_ohe = pd.DataFrame()
            new_sub_ohe["patient"] = all_subject
            new_sub_ohe["agg_ohe"] = all_agg_ohe
            k_fold = IterativeStratification(n_splits=config.param_split.fold, order=1)
            all_split_df = []
            for train, test in k_fold.split(new_sub_ohe["patient"].values, np.stack(new_sub_ohe["agg_ohe"].values)):
                test_df = data.loc[data['patient'].isin(list(new_sub_ohe.iloc[test].subject_id))]
                all_split_df.append(test_df)
            
    # ------------------------------------------------------------------------------------------
    # Hold-one-out
    elif config.param_split.type=='hoo':
        df = pd.read_pickle(config.path.processed_df_path)
        df["part"] = df.apply(lambda row: '_'.join(row.patient.split('_')[:2]), axis=1)
        # df.columns = ['subject_id', 'trial', 'sbp', 'dbp', 'sfppg', 'abp',  'part']
        if config.param_split.is_mix:
            ts_df = df.sample(frac=0.2, random_state=100)
            exclude = df.index.isin(ts_df.index.to_list())
            df = df[~exclude]
            val_df = df.sample(frac=0.2, random_state=100)
            exclude = df.index.isin(val_df.index.to_list())
            tr_df = df[~exclude]
        else:
            # Separate tr and val/ts by partitions
            tr_df = df[df.part.isin(config.param_split.tr_part)]
            if config.param_split.val_part==config.param_split.ts_part: 
                # split 20% of data by records as validation 
                val_ts_df = df[df.part.isin(config.param_split.val_part)]

                # Split val and ts by records
                all_recs = list(val_ts_df.subject_id.unique())
                random.seed(config.param_split.random_state)
                val_recs = random.sample(all_recs, int(0.2*len(all_recs)))

                val_df = val_ts_df[val_ts_df.subject_id.isin(val_recs)]
                exclude = val_ts_df.index.isin(val_df.index.to_list())
                ts_df = val_ts_df[~exclude]
            else: 
                val_df = df[df.part.isin(config.param_split.val_part)]
                ts_df = df[df.part.isin(config.param_split.ts_part)]

        all_split_df = [ts_df, val_df, tr_df]
    joblib.dump(all_split_df, config.path.split_df_path)
#%%
if __name__=='__main__':
    from omegaconf import OmegaConf 
    config = OmegaConf.load('/sensorsbp/code/process/core/config/process_uci_feat_5s.yaml')
    data_splitting(config)

    data = joblib.load(config.path.split_df_path)
    cnt = 0
    for d in data:
        cnt+=d.shape[0]
    print('total amount', cnt)