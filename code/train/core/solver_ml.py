#%%
import os
import joblib
import numpy as np 

# Load loaders
from core.loaders import *
from core.utils import cal_metric
from core.solver import Solver

# Load model
from core.models import *

# Others
import mlflow as mf
import coloredlogs, logging
import warnings
warnings.filterwarnings("ignore")

coloredlogs.install()
logger = logging.getLogger(__name__)  

#%%
class SolverML(Solver):
    def _get_model(self):
        if self.config.exp.model_type == "lgb":
            model = lgbModel(self.config.param_model, self.config.exp.random_state)
        elif self.config.exp.model_type == "rf":
            model = rfModel(self.config.param_model, random_state=self.config.exp.random_state)
        elif self.config.exp.model_type == "svr":
            model = svrModel(self.config.param_model, random_state=self.config.exp.random_state)
        elif self.config.exp.model_type == "mlp":
            model = mlpModel(self.config.param_model, random_state=self.config.exp.random_state)
        elif self.config.exp.model_type == "ada":
            model = adaModel(self.config.param_model, random_state=self.config.exp.random_state)
        else:
            model = eval(self.config.exp.model_type)(self.config.param_model, random_state=self.config.exp.random_state)
        return model
        
    def get_cv_metrics(self, fold_errors, dm, model, mode="val"):
        target = self.config.param_loader.label # sbp, dbp

        if mode=='tr':
            loader = dm.train_dataloader()
        elif mode=='val':
            loader = dm.val_dataloader()
        elif mode=='ts':
            loader = dm.test_dataloader()

        bp_denorm = loader.dataset.bp_denorm
        x, y = loader.dataset.all_ppg, loader.dataset.all_label#.reshape(-1)

        # error
        if target==['SP', 'DP']:
            # mlp outputs SP and DP simultaneously
            # prediction
            pred_bp = model.evaluate(x)
            pred_sbp = bp_denorm(pred_bp[:,0], 'SP')
            pred_dbp = bp_denorm(pred_bp[:,1], 'DP')

            # gorund truth
            true_sbp = bp_denorm(y[:,0], 'SP')
            true_dbp = bp_denorm(y[:,1], 'DP')

            # naive
            naive_bp =  np.mean(dm.train_dataloader(is_print=False).dataset.all_label, axis=0)
            naive_sbp = bp_denorm(naive_bp[0], 'SP')
            naive_dbp = bp_denorm(naive_bp[1], 'DP')
        
            sbp_err = pred_sbp - true_sbp
            dbp_err = pred_dbp - true_dbp
            metrics = cal_metric({'SP': sbp_err, 'DP': dbp_err}, mode=mode)
            fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
            fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
            fold_errors[f"{mode}_SP_pred"].append(pred_sbp)
            fold_errors[f"{mode}_SP_label"].append(true_sbp)
            fold_errors[f"{mode}_SP_naive"].append([naive_sbp]*len(pred_bp))
            fold_errors[f"{mode}_DP_pred"].append(pred_dbp)
            fold_errors[f"{mode}_DP_label"].append(true_dbp)
            fold_errors[f"{mode}_DP_naive"].append([naive_dbp]*len(pred_bp))
        
        else:
            # prediction
            pred_bp = bp_denorm(model.evaluate(x), target)

            # gorund truth
            true_bp = bp_denorm(y, target)

            # naive
            naive_bp =  np.mean(dm.train_dataloader(is_print=False).dataset.all_label, axis=0)
            naive_bp = bp_denorm(naive_bp, target)

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
        target = self.config.param_loader.label # SP, DP
        if target==['SP', 'DP']:
            fold_errors_template = {"subject_id":[],
                                    "record_id": [],
                                    "SP_naive":[],
                                    "SP_pred":[],
                                    "SP_label":[],
                                    "DP_naive":[],
                                    "DP_pred":[],
                                    "DP_label":[]
                                    }
        else:
            fold_errors_template = {"subject_id":[],
                                    "record_id": [],
                                    f"{target}_naive":[],
                                    f"{target}_pred":[],
                                    f"{target}_label":[]
                                    }
        
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["tr", "val","ts"]}
        
        # =============================================================================
        # data module
        # =============================================================================
        dm = self._get_loader()

        
        all_split_df = joblib.load(self.config.exp.subject_dict)
        # Nested cv 
        for foldIdx in range(self.config.exp.N_fold):
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            train_df = all_split_df[foldIdx]['train']
            val_df = all_split_df[foldIdx]['val']
            test_df = all_split_df[foldIdx]['test']
            
            dm.setup_kfold(train_df, val_df, test_df)
            
            # Init model
            model = self._get_model()
           
            # trainer main loop
            mf.pytorch.autolog()
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                x = dm.train_dataloader().dataset.all_ppg
                y = dm.train_dataloader().dataset.all_label

                # model = ToyModel(self.config.param_model)
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
        if target==['SP', 'DP']:
            sbp_err = fold_errors[f"ts_SP_naive"] - fold_errors[f"ts_SP_label"] 
            dbp_err = fold_errors[f"ts_DP_naive"] - fold_errors[f"ts_DP_label"] 
            naive_metric = cal_metric({'SP':sbp_err, 'DP':dbp_err}, mode='nv')
            out_metric.update(naive_metric)
            for mode in ['tr', 'val', 'ts']:
                sbp_err = fold_errors[f"{mode}_SP_pred"] - fold_errors[f"{mode}_SP_label"] 
                dbp_err = fold_errors[f"{mode}_DP_pred"] - fold_errors[f"{mode}_DP_label"] 
                tmp_metric = cal_metric({'SP':sbp_err, 'DP':dbp_err}, mode=mode)
                out_metric.update(tmp_metric)
        else:  
            bp_err = fold_errors[f"ts_{target}_naive"] - fold_errors[f"ts_{target}_label"] 
            naive_metric = cal_metric({target:bp_err}, mode='nv')
            out_metric.update(naive_metric)
            for mode in ['tr', 'val', 'ts']:
                bp_err = fold_errors[f"{mode}_{target}_pred"] - fold_errors[f"{mode}_{target}_label"] 
                tmp_metric = cal_metric({target:bp_err}, mode=mode)
                out_metric.update(tmp_metric)

        return out_metric

