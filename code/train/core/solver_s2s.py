#%%
import os
import joblib
from shutil import rmtree
import pandas as pd
import numpy as np
from scipy.io import loadmat
from glob import glob

# Load loaders
from core.loaders import *
from core.utils import (get_nested_fold_idx, get_ckpt, compute_sp_dp, cal_metric, cal_statistics, mat2df)

# Load model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from core.models import *

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
        self.config = config 

    def _get_loader(self):
        if self.config.exp.loader=="waveform":
            return WavDataModule(self.config)
        elif self.config.exp.loader=="mabp":
            return MABPDataModule(self.config)
        elif self.config.exp.loader=="feature":
            return FeatDataModule(self.config)

    def _get_model(self, ckpt_path_abs=None):
        model = None
        if not ckpt_path_abs:
            if self.config.exp.model_type == "unet1d":
                model = Unet1d(self.config.param_model, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "ppgiabp":
                model = PPGIABP(self.config.param_model, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "vnet":
                model = Vnet1d(self.config.param_model, random_state=self.config.exp.random_state)
            else:
                model = eval(self.config.exp.model_type)(self.config.param_model, random_state=self.config.exp.random_state)
            return model
        else:
            if self.config.exp.model_type == "unet1d":
                model = Unet1d.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "ppgiabp":
                model = PPGIABP.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "vnet":
                model = Vnet1d.load_from_checkpoint(ckpt_path_abs)
            else:
                model = eval(self.config.exp.model_type).load_from_checkpoint(ckpt_path_abs)
            return model
    
    def get_cv_metrics(self, fold_errors, dm, model, outputs, mode="val"):
        if mode=='val':
            loader = dm.val_dataloader()
        elif mode=='test':
            loader = dm.test_dataloader()

        bp_denorm = loader.dataset.bp_denorm
        
        #--- Predict
        true_abps = bp_denorm(outputs["true_abp"].squeeze(1).numpy(), self.config, 'abp')
        pred_abps = bp_denorm(outputs["pred_abp"].squeeze(1).numpy(), self.config, 'abp')
        true_bps = outputs["true_bp"].numpy()
        if self.config.exp.model_type in ["unet1d","vnet"]:
            pred_bps = np.array([compute_sp_dp(p) for p in pred_abps])
        elif self.config.exp.model_type == "ppgiabp":
            mask = outputs['y_mask']==1
            pred_bps = np.array([[np.max(pred_abps[i][mask[i]]),np.min(pred_abps[i][mask[i]])] for i in range(len(pred_abps))])
            
        naive_bps =  np.mean(dm.train_dataloader().dataset._target_data, axis=0)
        
        #--- Evaluate
        err_dict = {}
        for i, tar in enumerate(['SP', 'DP']):
            tar_acrny = 'sbp' if tar=='SP' else 'dbp'
            pred_bp = pred_bps[:,i]
            true_bp = bp_denorm(true_bps[:,i], self.config, tar)
            naive_bp = bp_denorm(naive_bps[i], self.config, tar)

            # error
            err_dict[tar_acrny] = pred_bp - true_bp
            fold_errors[f"{mode}_{tar_acrny}_pred"].append(pred_bp)
            fold_errors[f"{mode}_{tar_acrny}_label"].append(true_bp)
            fold_errors[f"{mode}_{tar_acrny}_naive"].append([naive_bp]*len(pred_bp))
        
        fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
        fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
        fold_errors[f"{mode}_abp_true"].append(true_abps)
        fold_errors[f"{mode}_abp_pred"].append(pred_abps)
        
        metrics = cal_metric(err_dict, mode=mode)    
        
        return metrics
            
#%%
    def evaluate(self):
        fold_errors_template = {"subject_id":[],
                                "record_id": [],
                                "abp_true":[],
                                "abp_pred":[]}
        for tar in ['sbp', 'dbp']:
            fold_errors_template[f"{tar}_naive"] = []
            fold_errors_template[f"{tar}_pred"] = []
            fold_errors_template[f"{tar}_label"] = []
            
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["val","test"]}
        
        #--- Data module
        dm = self._get_loader()
        
        
        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]
        
        #--- Nested cv 
        self.config = cal_statistics(self.config, all_split_df)
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            test_df = pd.concat(np.array(all_split_df)[folds_test])
            
            dm.setup_kfold(train_df, val_df, test_df)

            #--- Init model
            model = self._get_model()
            early_stop_callback = EarlyStopping(**dict(self.config.param_early_stop))
            checkpoint_callback = ModelCheckpoint(**dict(self.config.logger.param_ckpt))
            lr_logger = LearningRateMonitor()
            trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ])

            #--- Trainer main loop
            mf.pytorch.autolog()
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                # train
                trainer.fit(model, dm)
                artifact_uri, ckpt_path = get_ckpt(mf.get_run(run_id=run.info.run_id))

                # load best ckpt
                ckpt_path_abs = str(Path(artifact_uri)/ckpt_path[0])
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
                
                # log metrics
                metrics = self.get_cv_metrics(fold_errors, dm, model, val_outputs, mode="val")
                metrics = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test")
                logger.info(f"\t {metrics}")
                mf.log_metrics(metrics)

            #--- Save to model directory
            os.makedirs(self.config.path.model_directory, exist_ok=True)
            trainer.save_checkpoint("{}/{}-fold{}-test_sp={:.3f}-test_dp={:.3f}.ckpt".format(
                                                                           self.config.path.model_directory,
                                                                           self.config.exp.exp_name,
                                                                           foldIdx, 
                                                                           metrics["test/sbp_mae"],
                                                                           metrics["test/dbp_mae"]))
        
        #--- compute final metric
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)
        
        for mode in ['val', 'test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'dbp']}
            tmp_metric = cal_metric(err_dict, mode=mode)
            out_metric.update(tmp_metric)
            
        return out_metric
    
    def test(self):
        results = {}
        fold_errors_template = {"subject_id":[],
                                "record_id": [],
                                "abp_true":[],
                                "abp_pred":[]}
        for tar in ['sbp', 'dbp']:
            fold_errors_template[f"{tar}_naive"] = []
            fold_errors_template[f"{tar}_pred"] = []
            fold_errors_template[f"{tar}_label"] = []
            
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["test"]}
        
        #--- Data module
        dm = self._get_loader()
        
        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]
        
        #--- Nested cv 
        self.config = cal_statistics(self.config, all_split_df)
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            test_df = pd.concat(np.array(all_split_df)[folds_test])
            
            dm.setup_kfold(train_df, val_df, test_df)

            #--- load trained model
            if 'param_trainer' in self.config.keys():
                trainer = MyTrainer(**dict(self.config.param_trainer))
            else:
                trainer = MyTrainer()
            ckpt_apth_abs = glob(f'{self.config.param_test.model_path}{foldIdx}' + '*.ckpt')[0]
            model = self._get_model(ckpt_path_abs=ckpt_apth_abs)
            model.eval()
            trainer.model = model
            
            #--- get test output
            test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True)
            metrics = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test")
            logger.info(f"\t {metrics}")

        #--- compute final metric
        results['fold_errors'] = fold_errors    
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)
        
        for mode in ['test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'dbp']}
            tmp_metric = cal_metric(err_dict, mode=mode)
            out_metric.update(tmp_metric)
        
        results['out_metric'] = out_metric 
        
        os.makedirs(os.path.dirname(self.config.param_test.save_path), exist_ok=True)
        joblib.dump(results, self.config.param_test.save_path)  
        
        print(out_metric)   