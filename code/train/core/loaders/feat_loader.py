#%%
import pandas as pd
import numpy as np
from core.utils import (global_denorm, glob_dez, glob_z, glob_demm, glob_mm, 
                        loc_dez, loc_z, loc_demm, loc_mm)

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
        dataset = sensorsLoader(config=self.config, 
                                data_df=data_df, 
                                mode=mode,
                                is_print=is_print)
        
        return DataLoader(dataset=dataset, batch_size=dataset.__len__(), shuffle=(mode=="train"))

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
        if self.config.exp.model_type=='mlp':
            ft_imp_SP = pd.read_pickle(self.config.exp.feat_importance.replace('-DP', '-SP')) 
            ft_imp_DP = pd.read_pickle(self.config.exp.feat_importance.replace('-SP', '-DP'))
            ft_total= ft_imp_SP.merge(ft_imp_DP,on='features')
            ft_total['importance'] = (ft_total['importance_x']+ft_total['importance_y'])/2
            df_imp = ft_total[['features','importance']].sort_values('importance',ascending=False)
        else:    
            file_path = self.config.exp.feat_importance 
            df_imp = pd.read_pickle(file_path)
        sorted_feat = df_imp.features.values
        len_features = len(sorted_feat)
        features = sorted_feat[:int(len_features*self.config.param_loader['rate_features'])]
        
        # get ppg
        all_ppg = self.data_df[features]
        if not self.config.exp.model_type in ['lgb', 'rf']:
            # since lightgbm package is able to handle NaN,
            # lgb and rf models imported from the package are excluded
            all_ppg = all_ppg.fillna(0).values
        
        self.all_ppg = np.array(all_ppg)
        self.all_label = np.stack(self.data_df[target].values)
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
