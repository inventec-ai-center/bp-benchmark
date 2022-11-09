#%%
import random
import torch
import numpy as np
from core.utils import (print_criterion, get_bp_pk_vly_mask,
                        glob_dez, glob_z, glob_demm, glob_mm, 
                        loc_dez, loc_z, loc_demm, loc_mm)

import pytorch_lightning as pl
from torch.utils.data import DataLoader

SEED = 0
def seed_worker(SEED):
    np.random.seed(SEED)
    random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)

class WavDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup_kfold(self, folds_train, folds_val, folds_test):
        self.folds_train = folds_train
        self.folds_val = folds_val
        self.folds_test = folds_test

    def _get_loader(self, data_df, mode, is_print=False):
        dataset = sensorsLoader(config=self.config, 
                                data_df=data_df, 
                                mode=mode,
                                is_print=is_print)

        batch_size = self.config.param_model.batch_size
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode=="train"), 
                          worker_init_fn=seed_worker)
        
    def train_dataloader(self, is_print=False):
        return self._get_loader(self.folds_train, "train", is_print=is_print)

    def val_dataloader(self, is_print=False):
        return self._get_loader(self.folds_val, "val", is_print=is_print)

    def test_dataloader(self, is_print=False):
        return self._get_loader(self.folds_test, "test", is_print=is_print)

#%%
class sensorsLoader():
    def __init__(self, config, data_df, mode="train", is_print=False):
        self.config = config
        self.data_df = data_df
        self.mode = mode
        self._normalization()
        self._get_signal_feature()
        amounts = len(self.all_ppg)
        sbps = self.bp_denorm(self._target_data[:,0], self.config, 'SP')
        dbps = self.bp_denorm(self._target_data[:,1], self.config, 'DP')

        if is_print:
            print("Loader length: ", amounts)
            print_criterion(sbps, dbps)
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
        all_ppg = []
        all_abp = []
        all_sbp = []
        all_dbp = []

        dummy = False
        # check if abp signal exist -> if not insert dummy
        if not 'abp_signal' in list(self.data_df.columns):
            self.data_df['abp_signal'] = self.data_df['signal']

        # get ppg
        all_ppg  = np.stack(self.data_df["signal"].values)
        all_ppg = np.array([self.ppg_norm(ppg, self.config, type='ppg') for ppg in all_ppg])
        all_ppg = np.expand_dims(np.array(all_ppg), axis=1).astype("float32")
       
        if self.config.param_loader.get('feat'):
            for feattype in self.config.param_loader.feat:
                if feattype=='vpg':
                    vpg = all_ppg[:, 1:] - all_ppg[:, :-1]
                    padding = np.zeros(shape=(vpg.shape[0], 1))
                    vpg = np.concatenate([padding, vpg], axis=-1)
                    vpg = np.expand_dims(np.array(vpg), axis=1)
                    all_ppg = np.concatenate([all_ppg, vpg], axis=1).astype("float32")
                if feattype=='apg':
                    apg = vpg[:, 1:] - vpg[:, :-1]
                    padding = np.zeros(shape=(apg.shape[0], 1))
                    apg = np.concatenate([padding, apg], axis=-1)
                    apg = np.expand_dims(np.array(apg), axis=1)
                    all_ppg = np.concatenate([all_ppg, apg], axis=1).astype("float32")
                if feattype=='demo':
                    age =  np.tile(self.data_df["norm_age"].values, (625,1,1))   
                    age = np.transpose(age, (2,1,0))
                    sex =  np.tile(self.data_df["sex"].values, (625,1,1))   
                    sex = np.transpose(sex, (2,1,0))
                    all_ppg = np.concatenate([all_ppg, sex, age], axis=1).astype("float32")
                if feattype=='hr':
                    age =  np.tile(self.data_df["norm_age"].values, (625,1,1))   
                    age = np.transpose(age, (2,1,0))
                    sex =  np.tile(self.data_df["sex"].values, (625,1,1))   
                    sex = np.transpose(sex, (2,1,0))
                    all_ppg = np.concatenate([all_ppg, sex, age], axis=1).astype("float32")
                if feattype=='ecg':
                    ecg = np.stack(self.data_df["ecg"].values)
                    ecg = np.array([self.ppg_norm(e, self.config, type='ppg') for e in ecg])
                    ecg = np.expand_dims(np.array(ecg), axis=1).astype("float32")
                    all_ppg = np.concatenate([all_ppg, ecg], axis=1).astype("float32")

            self.all_ppg = all_ppg           

        else:
            self.all_ppg = all_ppg
        
        all_abp = self.bp_norm(np.stack(self.data_df["abp_signal"].values), self.config, type="abp")

        all_sbp = self.bp_norm(np.stack(self.data_df["SP"].values).reshape(-1,1), self.config, type="SP")
        all_dbp = self.bp_norm(np.stack(self.data_df["DP"].values).reshape(-1,1), self.config, type="DP")
        
        self.all_abp = np.expand_dims(np.array(all_abp), axis=1).astype("float32")
        self._target_data = np.concatenate([all_sbp, all_dbp],axis=1).astype("float32")

        self.subjects = list(self.data_df['patient']) 
        self.records = list(self.data_df['trial']) 

    def __getitem__(self, index):
        # Non-signal data
        signal = {'ppg': self.all_ppg[index]}
        abp = self.all_abp[index]
        y = self._target_data[index]
        
        peakmask, vlymask = get_bp_pk_vly_mask(abp.reshape(-1))
        return signal, y, abp, peakmask, vlymask
