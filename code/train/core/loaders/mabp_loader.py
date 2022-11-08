#%%
import random
import torch
import numpy as np
from core.utils import (print_criterion, get_bp_pk_vly_mask,
                        glob_dez, glob_z, glob_demm, glob_mm, 
                        loc_dez, loc_z, loc_demm, loc_mm)

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from scipy.signal import savgol_filter

SEED = 0
def seed_worker(SEED):
    np.random.seed(SEED)
    random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)

class MABPDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup_kfold(self, folds_train, folds_val, folds_test):
        self.folds_train = folds_train
        self.folds_val = folds_val
        self.folds_test = folds_test

    def _get_loader(self, data_df, mode):
        dataset = MABPLoader(config=self.config, 
                                data_df=data_df, 
                                mode=mode)

        batch_size = self.config.param_model.batch_size
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode=="train"), 
                          worker_init_fn=seed_worker)
        
    def train_dataloader(self):
        return self._get_loader(self.folds_train, "train")

    def val_dataloader(self):
        return self._get_loader(self.folds_val, "val")

    def test_dataloader(self):
        return self._get_loader(self.folds_test, "test")

#%%
class MABPLoader(Dataset):
    def __init__(self, config, data_df, mode="train"):
        self.config = config
        self.data_df = data_df
        self.mode = mode
        self._normalization()
        self._get_signal_feature()
        
    def _normalization(self):
        if self.config.param_loader.ppg_norm=='loc_z':
            self.ppg_norm = loc_z
            self.ppg_denorm = loc_dez
        elif self.config.param_loader.ppg_norm=='loc_mm':
            self.ppg_norm = loc_mm 
            self.ppg_denorm = loc_demm   
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
        
        # PPG
        all_ppg = np.stack(self.data_df["signal"].values)
        all_ppg = np.array([self.ppg_norm(ppg, None, type='ppg') for ppg in all_ppg])
        all_ppg = np.expand_dims(np.array(all_ppg), axis=1).astype("float32")

        # VPG
        vpg = savgol_filter(all_ppg,7,3,mode='mirror',deriv=1).astype("float32") # Compute vpg
        vpg = np.stack([self.ppg_norm(s, None, type='ppg') for s in vpg])
        all_ppg = np.concatenate([all_ppg, vpg], axis=1).astype("float32")
        
        self.all_ppg = all_ppg
        
        # ABP and BP labels
        all_abp = self.bp_norm(np.stack(self.data_df["abp_signal"].values), self.config, type="abp")
        self.all_abp = np.expand_dims(np.array(all_abp), axis=1).astype("float32")
        
        all_sbp = self.bp_norm(np.stack(self.data_df["SP"].values).reshape(-1,1), self.config, type="SP")
        all_dbp = self.bp_norm(np.stack(self.data_df["DP"].values).reshape(-1,1), self.config, type="DP")
        self._target_data = np.concatenate([all_sbp, all_dbp],axis=1).astype("float32")
        
        self.subjects = list(self.data_df['patient']) 
        self.records = list(self.data_df['trial'])
        
        # Y (abp, masks) and mask
        abp_size = all_abp.shape[1]
        y_masks = np.zeros((self.data_df.shape[0], 4, abp_size), dtype='float32')
        self.mask = np.zeros((self.data_df.shape[0], abp_size), dtype='float32')
        
        for i in range(self.data_df.shape[0]):
            cl = self.data_df.class_limits.iloc[i]
            y_masks[i, 0, cl[2]:] = 1
            y_masks[i, 1, :cl[0]] = 1
            y_masks[i, 2, cl[0]:cl[1]] = 1
            y_masks[i, 3, cl[1]:cl[2]] = 1
            self.mask[i, :cl[3]] = 1
        
        y = np.expand_dims(all_abp, axis=1)
        self.y = np.concatenate([y, y_masks], axis=1).astype("float32")
        
    def __len__(self):
        return len(self.all_ppg)
    
    def __getitem__(self, index):
        return self.all_ppg[index], self.y[index], self.mask[index], self._target_data[index]
