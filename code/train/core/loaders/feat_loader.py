#%%
import os
os.chdir('/sensorsbp/code/train')
import joblib
import random
import pandas as pd
import numpy as np
from core.utils import (print_criterion, get_bp_pk_vly_mask, get_nested_fold_idx,
                        glob_dez, glob_z, glob_demm, glob_mm, 
                        loc_dez, loc_z, loc_demm, loc_mm)
from random import sample
from omegaconf import OmegaConf

import pytorch_lightning as pl
from torch.utils.data import DataLoader

class FeatDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup_kfold(self, folds_train, folds_val, folds_test):
        self.folds_train = folds_train
        self.folds_val = folds_val
        self.folds_test = folds_test

    def _get_loader(self, data_df, mode, is_print=False):
        if self.config.param_loader.phase_match:
            dataset = sensorsLoader(config=self.config, 
                                    data_df=data_df, 
                                    mode=mode,
                                    phase_match=self.config.param_loader.phase_match,
                                    filtered=self.config.param_loader.filtered,
                                    is_print=is_print)
        else:
            print("Get your aligned dataset ready!")

        return DataLoader(dataset=dataset, batch_size=dataset.__len__(), shuffle=(mode=="train"))

    def train_dataloader(self, is_print=False):
        return self._get_loader(self.folds_train, "train", is_print=is_print)

    def val_dataloader(self, is_print=False):
        return self._get_loader(self.folds_val, "val", is_print=is_print)

    def test_dataloader(self, is_print=False):
        return self._get_loader(self.folds_test, "test", is_print=is_print)

#%%
class sensorsLoader():
    def __init__(self, config, data_df, mode="train", phase_match=True, filtered=True, is_print=False):
        self.config = config
        self.data_df = data_df
        self.filtered = filtered
        self.phase_match = phase_match
        self.mode = mode
        self._normalization()
        self._get_signal_feature()
        amounts = len(self.all_ppg)
        # sbps = self.bp_denorm(self._target_data[:,0], self.config, 'sbp')
        # dbps = self.bp_denorm(self._target_data[:,1], self.config, 'dbp')

        if is_print:
            print("Loader length: ", amounts)
            # print_criterion(sbps, dbps)
            print("PPG min {:.2f}, max {:.2f}".format(np.min(self.all_ppg), np.max(self.all_ppg)))
            print(f"ppg shape: {self.all_ppg.shape}")
    

    def __len__(self):
        return len(self.all_ppg)

    def _normalization(self):
        if self.config.param_loader.ppg_norm=='loc_z':
            self.ppg_norm = loc_z
            self.ppg_denorm = loc_dez
        elif self.config.param_loader.ppg_norm=='loc_mm':
            self.ppg_norm = loc_mm 
            self.ppg_denorm = loc_dez   
        elif self.config.param_loader.ppg_norm=='glob_z':
            self.ppg_norm = glob_z   
            self.ppg_denorm = glob_dez 
        elif self.config.param_loader.ppg_norm=='glob_mm':
            self.ppg_norm = glob_mm  
            self.ppg_denorm = glob_demm  
        
        if self.config.param_loader.bp_norm=='loc_z':
            self.bp_norm = loc_z
            self.bp_denorm = loc_dez
        elif self.config.param_loader.bp_norm=='loc_mm':
            self.bp_norm = loc_mm
            self.bp_denorm = loc_demm
        elif self.config.param_loader.bp_norm=='glob_z':
            self.bp_norm = glob_z
            self.bp_denorm = glob_dez
        elif self.config.param_loader.bp_norm=='glob_mm':
            self.bp_norm = glob_mm
            self.bp_denorm = glob_demm

    def _get_signal_feature(self):
        all_ppg = []
        all_abp = []
        all_sbp = []
        all_dbp = []

        # get ppg
        if self.phase_match:
            if self.filtered:
                all_ppg  = np.stack(self.data_df["sfppg"].values)
            else:
                all_ppg  = np.stack(self.data_df["srppg"].values)
        else:
            if self.filtered:
                all_ppg  = np.stack(self.data_df["fppg"].values)
            else:
                all_ppg  = np.stack(self.data_df["rppg"].values)

        self.all_ppg = np.array([self.ppg_norm(ppg, self.config, type='ppg') for ppg in all_ppg])
        self.all_label = self.bp_norm(np.stack(self.data_df[self.config.param_loader.label].values).reshape(-1,1), self.config, type="sbp")
        
        self.subjects = list(self.data_df['subject_id']) 
        self.records = list(self.data_df['trial']) 

    def __getitem__(self, index):
        # Non-signal data
        ppg = self.all_ppg[index]
        y = self.all_label[index]
        
        return ppg, y

#%%
if __name__=='__main__':
    import joblib
    from core.utils import cal_statistics
    config = OmegaConf.load('/sensorsbp/code/train/core/config/unet_sensors_5s.yaml')
    all_split_df = joblib.load(config.exp.subject_dict)
    config= cal_statistics(config, all_split_df)


    for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(5)):
        if foldIdx==0:  break
    train_df = pd.concat(np.array(all_split_df)[folds_train])
    val_df = pd.concat(np.array(all_split_df)[folds_val])
    test_df = pd.concat(np.array(all_split_df)[folds_test])

    dm = FeatDataModule(config)
    dm.setup_kfold(train_df, val_df, test_df)
    # dm.train_dataloader()
    # dm.val_dataloader()
    # dm.test_dataloader()

    # ppg, y, abp, peakmask, vlymask = next(iter(dm.test_dataloader().dataset))
    for i, (ppg, y, abp) in enumerate(dm.test_dataloader()):
        print(ppg.shape)