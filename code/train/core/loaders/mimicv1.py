import sys
import json
PATHS = json.load(open("../../paths.json"))
for k in PATHS: 
    sys.path.append(PATHS[k])

import os
import numpy as np
import pandas as pd 
import pickle as pkl
import time
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

# from core import signal_processing
# from core.signal_processing.utils import load_wfdb, to_segments, get_bp_labels, is_flat, global_norm, global_denorm, get_hyp, waveform_norm, tic, toc
# from core.signal_processing.clean import SQI as sqi
# from core.signal_processing.extract import PPG

class MIMICv1(Dataset):
    _subjects = ["484","225","437","216","417","284", "438",
                 "471","213","439","237","240","446",
                 "281","476","224","226","427","482",
                 "485","443","276","452","472","230"]

    _N_folds = 5
    _cal_pool_size = 10  # This is a static number 
    _tabular_features = ["ptt","sbp","dbp","hr"]
    _ppg_features = ["ppg","vpg","apg"]
    _ecg_features = ["ecg"]
    _subject_folds = {}
    
    def __init__(self, db_path, features, target,
                 split_type, subjects=None, folds=None,
                 mode="train", N_cal_points=0, random_state=100):       
        self.db_path = db_path
        self.features = features
        self.target = target
        self.split_type = split_type
        self.subjects = subjects
        self.folds = folds
        self.mode = mode 
        self.N_cal_points = N_cal_points
        self.random_state = random_state

        # Ensure reproducibility
        self.random_state = random_state
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # Separate features
        self.tabular_features = []
        self.ppg_features = []
        self.ecg_features = []
        for f in features:
            if f in self._tabular_features:
                self.tabular_features.append(f)
            elif f in self._ppg_features:
                self.ppg_features.append(f)
                
            elif f in self._ecg_features:
                self.ecg_features.append(f)
            else:
                print("[WARNING] Feature {} is not supported, skipping.".format(f))

        # Load annotation and prepare splits
        self._load_anno(db_path, split_type)               
        self._filter(subjects, folds, mode, N_cal_points)
        self._norm()
        
        # Mapping between subject and folds
        for k in range(self._N_folds):
            if k not in self._subject_folds:
                self._subject_folds[k] = self._subjects

        # self.data.loc function is super slow, create runtime buffer for faster run
        self.data_tabular = self.data[self.tabular_features].values.astype("float32")
        self.data_target = self.data[self.target].values.astype("float32")
        self.data_signal_path = self.data["signal_path"].values


    def _load_anno(self, db_path, split_type):
        kfold = KFold(self._N_folds)
        loader_tag = self._get_loader_tag()
        cache_path = self.db_path+"/db_cache/{}.csv".format(loader_tag)

        if os.path.exists(cache_path):
            self.anno = pd.read_csv(cache_path)

            # Note the subject-folds
            if split_type == "cross-subject":
                subjects = np.array(self._subjects)
                for k, (train, test) in enumerate(kfold.split(self._subjects)):
                    for s in subjects[test]:
                        self._subject_folds[k] = subjects[test]
            return 
            
        self.anno = []
        if split_type == "in-subject":
            for s in self._subjects:
                # Load the .csv
                df = pd.read_csv(db_path+"{}.csv".format(s))  
                
                # Apply filtering over the rows, the following thresholds (obtained through eye-balling)
                mask = (df.sbp_std < 10) \
                           & ~(pd.isnull(df.ptt)) \
                           & (df.dbp_std < 5) \
                           & (df.ptt.between(100,600)) \
                           & (df.sbp_mean.between(60,200)) \
                           & (df.dbp_mean.between(30,160))
                df = df[mask].reset_index(drop=True)
                
                # Annotate the folds
                for k, (train, test) in enumerate(kfold.split(df)):
                    df.loc[test, "k"] = k  # Assign the fold label
                self.anno.append(df)
            
        elif split_type == "cross-subject":
            subjects = np.array(self._subjects)
            for k, (train, test) in enumerate(kfold.split(self._subjects)):
                for s in subjects[test]:
                    # Load the .csv
                    df = pd.read_csv(db_path+"{}.csv".format(s))
                    
                    # Apply filtering over the rows, the following thresholds (obtained through eye-balling)
                    mask = (df.sbp_std < 10) \
                               & ~(pd.isnull(df.ptt)) \
                               & (df.dbp_std < 5) \
                               & (df.ptt.between(100,600)) \
                               & (df.sbp_mean.between(60,200)) \
                               & (df.dbp_mean.between(30,160))
                    df = df[mask].reset_index(drop=True)
                    
                    df["k"] = k  # Assign the fold label
                    self.anno.append(df)
                self._subject_folds[k] = subjects[test]
            
        else:
            raise RuntimeError("Error split_type {} not supported, please choose between in-subject or cross-subject".format(split_type))
    
        # Consolidate
        self.anno = pd.concat(self.anno)        
        self.anno["sbp"] = self.anno["sbp_mean"]
        self.anno["dbp"] = self.anno["dbp_mean"]

        # self.anno = self.anno[["subject_id", "rec_date","rec_time", "signal_path", "k", "sbp_mean", "dbp_mean", "sbp", "dbp"] + self.tabular_features]

        # Store it as cache
        self.anno.to_csv(cache_path, index=False)

    def _filter(self, subjects, folds, mode, N_cal_points):
        # When subjects or none are not defined, the default value is all
        if subjects is None: subjects = self._subjects
        if folds is None: folds = list(np.arange(5))    
        
        # Filtering based on the subjects and folds
        mask_subjects = self.anno.subject_id.isin(subjects)
        mask_folds = self.anno.k.isin(folds)
        mask = mask_subjects & mask_folds
            
        # Apply the filter
        self.data = self.anno[mask].sort_values(["rec_date", "rec_time"]).reset_index(drop=True)

        # Special split for calibration or testing
        if mode == "cal":
            pool = self.data[:self._cal_pool_size]
            self.data = pool.sample(n=N_cal_points, random_state=self.random_state).reset_index(drop=True)
        elif mode == "test":
            self.data = self.data[self._cal_pool_size:].reset_index(drop=True)

    def _get_loader_tag(self):
        features = "-".join(self.features)
        target = "-".join(self.target)

        train_folds = ""
        test_folds = ""

        # Get folds that are not currently observed
        current_folds = np.array(self.folds)
        complement_folds = np.array([f for f in range(self._N_folds) if f not in self.folds])

        if self.mode == "train":
            train_folds = "".join(current_folds.astype("str"))
            test_folds = "".join(complement_folds.astype("str"))
        else:
            train_folds = "".join(complement_folds.astype("str"))
            test_folds =  "".join(current_folds.astype("str"))

        return f"{features}_{self.split_type}_TR{train_folds}_TE{test_folds}_{self.random_state}"

    def _load_scales(self):
        loader_tag = self._get_loader_tag()
        scale_path = self.db_path+"scales/{}.pkl".format(loader_tag)

        scales = {}
        if os.path.exists(scale_path) == False:
            # Compute the scales for each of the features
            for f in self._tabular_features:
                mean_data = self.data[f].mean()
                std_data = self.data[f].std()
                min_data = self.data[f].min()
                max_data = self.data[f].max()

                scales[f] = (mean_data, std_data, min_data, max_data)

            # Store the scales
            pkl.dump(scales, open(scale_path, "wb"))
        else:
            # Load existing scalers
            scales = pkl.load(open(scale_path, "rb"))
        return scales

    def _norm(self):
        scales = self._load_scales()
        for f in self._tabular_features:
            if f in ["ptt","sbp","dbp"]:
                self.data[f] = global_norm(self.data[f].values, f)
            else:
                mean_data, std_data, min_data, max_data = scales[f]
                self.data[f] = (self.data[f] - min_data) / max_data  # MinMax
                # self.data[f] = (self.data[f] - mean_data) / std_data  # Standard
            
    def _get_signal_feature(self, signal_path, signal_type, feature=None):
        data = np.load(signal_path, allow_pickle=True)
        data = data[signal_type]

        if feature is not None:
            if signal_type == "ecg":
                ecg = ECG(data, fs=125)
            elif signal_type == "ppg":
                ppg = PPG(data, fs=125)            
                if feature == "vpg": data = ppg.vpg()
                elif feature == "apg": data = ppg.apg()
                    
        data = waveform_norm(data)
        return data.reshape([1, -1])
    def __len__(self):
        N = self.data.shape[0]
        return N

    def __getitem__(self, index):
        x_data = self.data_tabular[index]
        y = self.data_target[index]

        # Extract the signal features
        signal_path = self.data_signal_path[index]
        x_signal = []
        for f in self.ppg_features:
            out_feature = self._get_signal_feature(signal_path, signal_type="ppg", feature=f)
            x_signal.append(out_feature)
        if len(x_signal) > 0: x_signal = np.concatenate(x_signal, axis=0)
        else: x_signal = np.zeros((1,3750))

        return x_data, x_signal, y        


# Look into
# - RF Regressor (PTT)
# - Linear Regression
# - MLP
# - CNN
# - CNN + LSTM (Paper)
# - Handcraft PPG temporal features
# - Handcraft PPG frequency feature

        
# random_state = 100

# dataset = Mimicv1("/workspace/bp-measurement/src/notebooks/mimic1-anno/",
#                 features=["ptt","ppg","vpg","apg"],
#                 target=["sbp"],
#                 split_type="in-subject",subjects=["484"], folds=None,
#                 mode="train", n_cal_points=2, random_state=random_state)

# loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

# for x_data, x_signal, y in loader:
#     print(x_data.shape, x_signal.shape,  y.shape)
#     break
