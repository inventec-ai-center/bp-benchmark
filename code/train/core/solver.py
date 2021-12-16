#%%
import os
import joblib
from shutil import rmtree
import pandas as pd
import numpy as np 
from omegaconf import OmegaConf

# Load loaders
from core.loaders.wav_loader import WavDataModule
from core.utils import (get_nested_fold_idx, get_ckpt, compute_sp_dp, cal_metric, cal_statistics, log_config)

# Load model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor
from core.models import *
import torch

# Others
import mlflow as mf
import coloredlogs, logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

coloredlogs.install()
logger = logging.getLogger(__name__)  

#%%
class Solver:
    DEFAULTS = {}   
    def __init__(self, config):
        # Initialize
        # self.config_file = config_file
        # self.config = OmegaConf.load(self.config_file)
        self.config = config 

    def _get_model(self, pos_weight=None, ckpt_path_abs=None):
        model = None
        if not ckpt_path_abs:
            if self.config.exp.model_type == "unet1d":
                model = Unet1d(self.config.param_model, random_state=self.config.exp.random_state)
            else:
                model = eval(self.config.exp.model_type)(self.config.param_model, random_state=self.config.exp.random_state)
            return model
        else:
            if self.config.exp.model_type == "unet1d":
                model = Unet1d.load_from_checkpoint(ckpt_path_abs)
            else:
                model = eval(self.config.exp.model_type).load_from_checkpoint(ckpt_path_abs)
            return model
    
    def get_cv_metrics(self, fold_errors, dm, model, outputs, mode="val"):
        if mode=='val':
            loader = dm.val_dataloader()
        elif mode=='test':
            loader = dm.test_dataloader()

        bp_denorm = loader.dataset.bp_denorm

        # ABP
        true_abp = bp_denorm(outputs["true_abp"].squeeze(1).numpy(), self.config, 'sbp')
        pred_abp = bp_denorm(outputs["pred_abp"].squeeze(1).numpy(), self.config, 'sbp')

        # prediction
        pred_bp = np.array([compute_sp_dp(p) for p in pred_abp])
        pred_sbp = pred_bp[:,0]
        pred_dbp = pred_bp[:,1]

        # gorund truth
        true_bp = outputs["true_bp"].numpy()
        true_sbp = bp_denorm(true_bp[:,0], self.config, 'sbp')
        true_dbp = bp_denorm(true_bp[:,1], self.config, 'dbp')

        # naive
        naive_bp =  np.mean(dm.train_dataloader(is_print=False).dataset._target_data, axis=0)
        naive_sbp = bp_denorm(naive_bp[0], self.config, 'sbp')
        naive_dbp = bp_denorm(naive_bp[1], self.config, 'dbp')

        # error
        sbp_err = pred_sbp - true_sbp
        dbp_err = pred_dbp - true_dbp


        metrics = model._cal_metric(torch.tensor(outputs["pred_abp"]), torch.tensor(outputs["true_abp"]))
        metrics = cal_metric(sbp_err, dbp_err, metric=metrics, mode=mode)
        

        fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
        fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
        fold_errors[f"{mode}_sbp_pred"].append(pred_sbp)
        fold_errors[f"{mode}_sbp_label"].append(true_sbp)
        fold_errors[f"{mode}_dbp_pred"].append(pred_dbp)
        fold_errors[f"{mode}_dbp_label"].append(true_dbp)
        fold_errors[f"{mode}_abp_true"].append(true_abp)
        fold_errors[f"{mode}_abp_pred"].append(pred_abp)
        fold_errors[f"{mode}_sbp_naive"].append([naive_sbp]*len(pred_bp))
        fold_errors[f"{mode}_dbp_naive"].append([naive_dbp]*len(pred_bp))
        
        return metrics
            
#%%
    def evaluate(self):
        fold_errors_template = {"subject_id":[],
                                "record_id": [],
                                "sbp_naive":[],
                                "sbp_pred":[],
                                "sbp_label":[],
                                "dbp_naive":[],
                                "dbp_pred":[],
                                "dbp_label":[],
                                "abp_true":[],
                                "abp_pred":[]}
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["val","test"]}
        # =============================================================================
        # data module
        # =============================================================================
        dm = WavDataModule(self.config)


        # Nested cv 
        all_split_df = joblib.load(self.config.exp.subject_dict)
        self.config = cal_statistics(self.config, all_split_df)
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(5)):
            # if foldIdx==1:  break
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            test_df = pd.concat(np.array(all_split_df)[folds_test])
            
            dm.setup_kfold(train_df, val_df, test_df)

            # Init model
            model = self._get_model()
            early_stop_callback = EarlyStopping(**dict(self.config.param_early_stop))
            checkpoint_callback = ModelCheckpoint(**dict(self.config.logger.param_ckpt))
            lr_logger = LearningRateMonitor()
            if self.config.get("param_swa"):
                swa_callback = StochasticWeightAveraging(**dict(self.config.param_swa))
                trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ])
            else:
                trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ])

            # trainer main loop
            mf.pytorch.autolog()
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                # train
                trainer.fit(model, dm)
                print("run_id", run.info.run_id)
                artifact_uri, ckpt_path = get_ckpt(mf.get_run(run_id=run.info.run_id))

                # load best ckpt
                ckpt_path_abs = str(Path(artifact_uri)/ckpt_path[0])
                # if ":" in ckpt_path_abs:
                #     ckpt_path_abs = ckpt_path_abs.split(":",1)[1]
                model = self._get_model(ckpt_path_abs=ckpt_path_abs)

                # evaluate
                val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False)
                test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True)

                # save updated model
                trainer.model = model
                trainer.save_checkpoint(ckpt_path_abs)

                # clear redundant mlflow models (save disk space)
                redundant_model_path = Path(artifact_uri)/'model'
                if redundant_model_path.exists(): rmtree(redundant_model_path)

                metrics = self.get_cv_metrics(fold_errors, dm, model, val_outputs, mode="val")
                metrics = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test")
                logger.info(f"\t {metrics}")
                mf.log_metrics(metrics)
                # log_config(self.config_file)

            # Save to model directory
            os.makedirs(self.config.path.model_directory, exist_ok=True)
            trainer.save_checkpoint("{}/{}-fold{}-test_sp={:.3f}-test_dp={:.3f}.ckpt".format(
                                                                           self.config.path.model_directory,
                                                                           self.config.exp.exp_name,
                                                                           foldIdx, 
                                                                           metrics["test/sbp_mae"],
                                                                           metrics["test/dbp_mae"]))

        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        sbp_err = fold_errors["test_sbp_naive"] - fold_errors["test_sbp_label"] 
        dbp_err = fold_errors["test_dbp_naive"] - fold_errors["test_dbp_label"] 
        naive_metric = cal_metric(sbp_err, dbp_err, mode='nv')
        out_metric.update(naive_metric)

        sbp_err = fold_errors["val_sbp_pred"] - fold_errors["val_sbp_label"] 
        dbp_err = fold_errors["val_dbp_pred"] - fold_errors["val_dbp_label"] 
        val_metric = cal_metric(sbp_err, dbp_err, mode='val')
        out_metric.update(val_metric)

        sbp_err = fold_errors["test_sbp_pred"] - fold_errors["test_sbp_label"] 
        dbp_err = fold_errors["test_dbp_pred"] - fold_errors["test_dbp_label"] 
        test_metric = cal_metric(sbp_err, dbp_err, mode='test')
        out_metric.update(test_metric)
        
        return out_metric
