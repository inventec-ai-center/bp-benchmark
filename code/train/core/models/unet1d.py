#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_pl import Regressor
from mdnc import mdnc


class Unet1d(Regressor):
    def __init__(self, param_model, random_state=0):
        super(Unet1d, self).__init__(param_model, random_state)

        self.model = mdnc.modules.conv.UNet1d(param_model.output_channel,
                                                list(param_model.layers),
                                                in_planes=param_model.input_size,
                                                out_planes=1) 
        

#%%
if __name__=='__main__':
    from omegaconf import OmegaConf
    import pandas as pd
    import numpy as np
    import joblib
    import os
    os.chdir('/sensorsbp/code/train')
    from core.loaders.wav_loader import WavDataModule
    from core.utils import get_nested_fold_idx, cal_statistics
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import LearningRateMonitor
    from core.models.trainer import MyTrainer

    config = OmegaConf.load('/sensorsbp/code/train/core/config/unet_sensors_12s.yaml')
    all_split_df = joblib.load(config.exp.subject_dict)
    config = cal_statistics(config, all_split_df)
    for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(5)):
        if foldIdx==0:  break
    train_df = pd.concat(np.array(all_split_df)[folds_train])
    val_df = pd.concat(np.array(all_split_df)[folds_val])
    test_df = pd.concat(np.array(all_split_df)[folds_test])

    dm = WavDataModule(config)
    dm.setup_kfold(train_df, val_df, test_df)
    # dm.train_dataloader()
    # dm.val_dataloader()
    # dm.test_dataloader()
    
    # init model
    model = Unet1d(config.param_model)
    early_stop_callback = EarlyStopping(**dict(config.param_early_stop))
    checkpoint_callback = ModelCheckpoint(**dict(config.logger.param_ckpt))
    lr_logger = LearningRateMonitor()
    
    trainer = MyTrainer(**dict(config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ])

    # trainer main loop
    trainer.fit(model, dm)