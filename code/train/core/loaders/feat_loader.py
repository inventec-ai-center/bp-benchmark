#%%
import joblib
import random
import pandas as pd
import numpy as np
from core.utils import (global_denorm, print_criterion, get_bp_pk_vly_mask, get_nested_fold_idx,
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

        if is_print:
            print("Loader length: ", amounts)
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
        else:
            self.ppg_norm = None
            self.ppg_denorm = None 
        
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
        else:
            self.bp_norm = None 
            self.bp_denorm = global_denorm

    def _get_signal_feature(self):
        target = self.config.param_loader.label

        # select feature by importance
        file_path = self.config.exp.feat_importance if target=='SP' else self.config.exp.feat_importance.replace('-SP', '-DP')
        df_imp = pd.read_pickle(file_path)
        sorted_feat = df_imp.features.values
        len_features = len(sorted_feat)
        features = sorted_feat[:int(len_features*self.config.param_loader['rate_features'])]
        
        # get ppg
        # all_ppg = self.data_df.loc[:, ~self.data_df.columns.isin(['patient','trial','SP', 'DP','part'])]
        all_ppg = self.data_df[features]
        all_ppg = all_ppg.fillna(0).values
        
        self.all_ppg = np.array(all_ppg)
        self.all_label = np.stack(self.data_df[target].values).reshape(-1,1)
        
        if not self.ppg_norm is None:
            self.all_ppg = np.array([self.ppg_norm(ppg, self.config, type='ppg') for ppg in self.all_ppg])
        if not self.bp_norm is None:
            self.all_label = self.bp_norm(self.all_label, self.config, type=target)
        
        self.subjects = list(self.data_df['patient']) 
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
    config = OmegaConf.load('/sensorsbp/code/train/core/config/hydra/toyml_uci_5s.yaml')
    all_split_df = joblib.load(config.exp.subject_dict)
    config= cal_statistics(config, all_split_df)


    for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(3)):
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
    for i, (ppg, y) in enumerate(dm.test_dataloader()):
        print(ppg.shape)