import pandas as pd
from core.signal_processing.extract import PPG
from core.signal_processing.clean import SQI as sqi
from core.signal_processing.utils import (load_wfdb, to_segments, get_bp_labels, 
                                     is_flat, global_norm, global_denorm, 
                                     get_hyp, waveform_norm, tic, toc)
from core import signal_processing
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch
import time
import pickle as pkl
import numpy as np
import os
import sys
import json


class PassiveMIMICv1(Dataset):
    _tab_features = ["ptt", "hr", "sbp", "dbp"]
    _add_tab_feature = {"ppg_histogram_up": 5, "ppg_histogram_down": 10,
                        "vpg_histogram_up": 5, "vpg_histogram_down": 10,
                        "apg_histogram_up": 5, "apg_histogram_down": 10,
                        "ppg_fft_peaks_heights": 5, "ppg_fft_peaks_neighbor_avgs": 5,
                        "vpg_fft_peaks_heights": 5, "vpg_fft_peaks_neighbor_avgs": 5,
                        "apg_fft_peaks_heights": 5, "apg_fft_peaks_neighbor_avgs": 5,
                        "usdc": 6, "dsdc": 9}

    _cat_features = ["hod", "gender"]
    _sig_features = ["ecg", "ppg", "apg", "vpg","cycles","cycle_stats","cycle_mean"]
    _cal_pool_size = 10
    PATHS = json.load(open("../../paths.json"))

    def __init__(self, anno_data, features, target,
                 split_type, subjects, folds,
                 mode="train",
                 N_cal_points=0,
                 N_folds=5,
                 random_state=100):
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Copy values from parameters
        self.anno_data = anno_data.copy()
        self.N_folds = N_folds
        self.target = target
        self.mode = mode
        self.folds = folds
        self.subjects = subjects
        self.split_type = split_type
        self.N_cal_points = N_cal_points
        self.random_state = random_state

        # Process anno_data
        self._build_features(features)
        self._compute_scales()
        self._normalize()
        self._filter(subjects, folds)

        # Iterable arrays
        self.tab_data = self.data[self.tab_features].values.astype("float32")
        # Categorical data needs to consecutive and represented as LongInt
        self.cat_data = self.data[self.cat_features].values.astype("int")
        self.data_signal_path = self.data["signal_path"].values
        for i in range(self.data_signal_path.shape[0]):
            self.data_signal_path[i] = self.data_signal_path[i].replace("processed","extracted")
        self.target_data = self.data[self.target].values.astype("float32")

        # Caching especially for machines that are super slow in terms of cpu processing
        self.x_signal_cache = {}
        

    def _build_features(self, features):
        # Include additional features
        for item in self._add_tab_feature:
            N = self._add_tab_feature[item]
            add_features = [item+"_{}".format(x) for x in range(N)]
            self._tab_features = self._tab_features + add_features

        self.tab_features = []
        self.cat_features = []
        self.sig_features = []

        for f in features:
            if f in self._tab_features:
                self.tab_features.append(f)
            elif f in self._cat_features:
                self.cat_features.append(f)
            elif f in self._sig_features:
                self.sig_features.append(f)
            elif f in self._add_tab_feature:
                N = self._add_tab_feature[f]
                add_features = [f+"_{}".format(x) for x in range(N)]
                self.tab_features = self.tab_features + add_features

    def _get_folds(self):
        available_folds = np.arange(self.N_folds)

        curr_folds = np.array(self.folds)
        comp_folds = np.array(
            [f for f in available_folds if f not in self.folds])

        if self.mode == "train":
            return curr_folds, comp_folds
        else:
            return comp_folds, curr_folds

    def _compute_scales(self):
        # Compute the scales based on the train folds
        train_folds, test_folds = self._get_folds()

        # Choose folds based on split_type
        if self.split_type == "in-subject":
            self.anno_data["k"] = self.anno_data["IS_fold"]
        else:
            self.anno_data["k"] = self.anno_data["XS_fold"]

        # Get the train_df
        train_df = self.anno_data[self.anno_data.k.isin(train_folds)]

        scales = {}
        for f in self._tab_features:
            # These three columns are special bcs they have global range
            if f in ["ptt", "sbp", "dbp"]:
                continue

            # Other tabular columns
            min_stat = train_df[f].min()
            max_stat = train_df[f].max()
            mean_stat = train_df[f].mean()
            std_stat = train_df[f].std()
            scales[f] = (min_stat, max_stat, mean_stat, std_stat)
        self.scales = scales

    def _normalize(self):
        for f in self._tab_features:
            # These three columns are special bcs they have global range
            if f in ["ptt", "sbp", "dbp"]:
                self.anno_data[f] = global_norm(self.anno_data[f].values, f)
            else:
                min_stat, max_stat, mean_stat, std_stat = self.scales[f]
                self.anno_data[f] = (self.anno_data[f] - min_stat) / max_stat

    def _filter(self, subjects, folds):
        '''
        Processed anno_data into data by filtering the subjects
        '''
        # When subjects or none are not defined, the default value is all
        if subjects is None:
            subjects = self.anno_data.subject_id.unique()
        if folds is None:
            folds = list(np.arange(5))

        # Filtering based on the subjects and folds
        mask_subjects = self.anno_data.subject_id.isin(subjects)
        mask_folds = self.anno_data.k.isin(folds)
        mask = mask_subjects & mask_folds

        # Apply the filter
        self.data = self.anno_data[mask].sort_values(["rec_date", "rec_time"]).reset_index(drop=True)

        N = self.data.shape[0]
         
        
        # Special split for calibration or testing
        if self.mode == "cal":
            pool = self.data[:self._cal_pool_size]
            self.data = pool.sample(
                n=self.N_cal_points, random_state=self.random_state).reset_index(drop=True)
        elif self.mode == "test":
            self.data = self.data[self._cal_pool_size:].reset_index(drop=True)

    def _get_signal_feature(self, signal_path, feature):
        # change the root to current database root
        if signal_path.startswith("/training_db"): 
            signal_path = signal_path.replace('/training_db', self.PATHS['DB_PATH'])
        signal_data = np.load(signal_path, allow_pickle=True)
        
        if feature == "cycles": return signal_data[feature]
        elif feature == "cycle_stats": return signal_data[feature]
        elif feature == "cycle_mean": return signal_data["cycle_stats"][[0]]
        return signal_data[feature].reshape([1, -1])
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Non-signal data
        x_tabular = self.tab_data[index]
        x_categorical = self.cat_data[index]
        y = self.target_data[index]

        # Signal data
        if index not in self.x_signal_cache:
            
            signal_path = self.data_signal_path[index]

            # Stack the input features into a list
            x_signal = []
            for f in self.sig_features:
                x_signal.append(self._get_signal_feature(signal_path, f))

            # Stack them as numpy array
            if len(x_signal) > 0:
                x_signal = np.concatenate(x_signal, axis=0).astype("float32")
            else:
                x_signal = np.zeros((1, 3750)).astype("float32")

            # Store in local cache
            self.x_signal_cache[index] = x_signal
        else:
            x_signal = self.x_signal_cache[index]
        
        return x_tabular, x_categorical, x_signal, y
