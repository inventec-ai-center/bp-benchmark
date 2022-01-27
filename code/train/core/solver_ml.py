#%%
import os
import joblib
from shutil import rmtree
import pandas as pd
import numpy as np 
from omegaconf import OmegaConf

# Load loaders
from core.loaders import *
from core.utils import (get_nested_fold_idx, cal_metric, cal_statistics)
from core.solver import Solver

# Load model
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
class SolverML(Solver):
    def get_cv_metrics(self, fold_errors, dm, model, mode="val"):
        target = self.config.param_loader.label # sbp, dbp

        if mode=='tr':
            loader = dm.train_dataloader()
        elif mode=='val':
            loader = dm.val_dataloader()
        elif mode=='ts':
            loader = dm.test_dataloader()

        bp_denorm = loader.dataset.bp_denorm
        x, y = loader.dataset.all_ppg, loader.dataset.all_label.reshape(-1)

        # prediction
        pred_bp = bp_denorm(model.evaluate(x), self.config, target)

        # gorund truth
        true_bp = bp_denorm(y, self.config, target)

        # naive
        naive_bp =  np.mean(dm.train_dataloader(is_print=False).dataset.all_label, axis=0)
        naive_bp = bp_denorm(naive_bp, self.config, target)

        # error
        bp_err = pred_bp - true_bp

        metrics = cal_metric({target: bp_err}, mode=mode)
        
        fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
        fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
        fold_errors[f"{mode}_{target}_pred"].append(pred_bp)
        fold_errors[f"{mode}_{target}_label"].append(true_bp)
        fold_errors[f"{mode}_{target}_naive"].append([naive_bp]*len(pred_bp))
        
        return metrics

    def evaluate(self):
        target = self.config.param_loader.label # sbp, dbp
        fold_errors_template = {"subject_id":[],
                                "record_id": [],
                                "sbp_naive":[],
                                "sbp_pred":[],
                                "sbp_label":[],
                                "dbp_naive":[],
                                "dbp_pred":[],
                                "dbp_label":[]}
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["tr", "val","ts"]}
        # =============================================================================
        # data module
        # =============================================================================
        # dm = WavDataModule(self.config)
        dm = self._get_loader()

        # Nested cv 
        all_split_df = joblib.load(self.config.exp.subject_dict)
        self.config = cal_statistics(self.config, all_split_df)
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            if (self.config.exp.N_fold=='HOO') and (foldIdx==1):  break
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            test_df = pd.concat(np.array(all_split_df)[folds_test])
            
            dm.setup_kfold(train_df, val_df, test_df)

            # Init model
            model = self._get_model()
           
            # trainer main loop
            mf.pytorch.autolog()
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                x = dm.train_dataloader().dataset.all_ppg
                y = dm.train_dataloader().dataset.all_label.reshape(-1)

                model = ToyModel(self.config.param_model)
                model.fit(x,y)

                metrics = {}
                metrics.update(self.get_cv_metrics(fold_errors, dm, model, mode="tr"))
                metrics.update(self.get_cv_metrics(fold_errors, dm, model, mode="val"))
                metrics.update(self.get_cv_metrics(fold_errors, dm, model, mode="ts"))
                logger.info(f"\t {metrics}")
                mf.log_metrics(metrics)

            # Save to model directory
            os.makedirs(self.config.path.model_directory, exist_ok=True)
            
            
        # compute final metric
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items() if len(v)!=0 }
        bp_err = fold_errors[f"ts_{target}_naive"] - fold_errors[f"ts_{target}_label"] 
        naive_metric = cal_metric({target:bp_err}, mode='nv')
        out_metric.update(naive_metric)
        for mode in ['tr', 'val', 'ts']:
            bp_err = fold_errors[f"{mode}_{target}_pred"] - fold_errors[f"{mode}_{target}_label"] 
            tmp_metric = cal_metric({target:bp_err}, mode=mode)
            out_metric.update(tmp_metric)

        return out_metric


#%%
if __name__=='__main__':
    import os
    os.chdir('/sensorsbp/code/train')
    from omegaconf import OmegaConf

    config = OmegaConf.load("/sensorsbp/code/train/core/config/toyml_uci_5s.yaml")
    solver = SolverML(config)
    self = solver

    dm = FeatDataModule(config)
    all_split_df = joblib.load(self.config.exp.subject_dict)
    self.config = cal_statistics(self.config, all_split_df)
    for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
        if (self.config.exp.N_fold==3) and (foldIdx==1):  break
        train_df = pd.concat(np.array(all_split_df)[folds_train])
        val_df = pd.concat(np.array(all_split_df)[folds_val])
        test_df = pd.concat(np.array(all_split_df)[folds_test])
        
        dm.setup_kfold(train_df, val_df, test_df)
    
    for i, (x,y) in enumerate(dm.train_dataloader()):
        print(x.shape, y.shape)
    
    x = x.numpy()
    y = y.numpy()

    model = ToyModel(config.param_model)
    model.fit(x,y)
    pred = model.evaluate(x)
