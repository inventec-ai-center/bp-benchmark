#%%
import os
import joblib
import numpy as np 
import pandas as pd

# Load modules
from core.solver_s2s import Solver
from core.loaders import *
from core.models import *
from core.utils import (cal_metric, get_nested_fold_idx, norm_data)

# Others
import mlflow as mf
import coloredlogs, logging
import warnings
warnings.filterwarnings("ignore")

coloredlogs.install()
logger = logging.getLogger(__name__)  

#%%
class SolverF2l(Solver):
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
        target = self.config.param_loader.label # 'SP', 'DP'

        if mode=='tr':
            loader = dm.train_dataloader()
        elif mode=='val':
            loader = dm.val_dataloader()
        elif mode=='ts':
            loader = dm.test_dataloader()

        bp_denorm = loader.dataset.bp_denorm
        x, y = loader.dataset.all_ppg, loader.dataset.all_label

        #--- Predict
        m, n = y.shape[0], y.shape[1] if len(y.shape)==2 else 1
        naive =  np.mean(dm.train_dataloader(is_print=False).dataset.all_label, axis=0)
        naive = np.array([naive]) if isinstance(naive, np.float64) else naive
        pred = model.evaluate(x).reshape(m, n)
        y = y.reshape(m, n)
        
        #--- Evaluate
        # make target a list: ['SP'], ['DP'], ['SP', 'DP']
        if target=='SP' or target=='DP':
            target = [target]
        
        metrics = {}    
        for i, tar in enumerate(target):
            pred_bp = bp_denorm(pred[:, i], tar)
            true_bp = bp_denorm(y[:, i], tar)
            naive_bp = bp_denorm(naive[i], tar)
        
            # error
            bp_err = pred_bp - true_bp
            metrics.update(cal_metric({tar: bp_err}, mode=mode))
            fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
            fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
            fold_errors[f"{mode}_{tar}_pred"].append(pred_bp)
            fold_errors[f"{mode}_{tar}_label"].append(true_bp)
            fold_errors[f"{mode}_{tar}_naive"].append([naive_bp]*len(pred_bp))
            
        return metrics

    def evaluate(self):
        target = self.config.param_loader.label # SP, DP
        # make target a list: ['SP'], ['DP'], ['SP', 'DP']
        if target=='SP' or target=='DP':
            target = [target]
        
        fold_errors_template = {"subject_id":[],
                                "record_id": []}
        for tar in target:
            fold_errors_template[f"{tar}_naive"] = []
            fold_errors_template[f"{tar}_pred"] = []
            fold_errors_template[f"{tar}_label"] = []
            
        
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["tr", "val","ts"]}
        
        #--- data module
        dm = self._get_loader()

        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [pd.read_csv(f"{self.config.exp.subject_dict}_{i}.csv", float_precision='round_trip') for i in range(self.config.exp.N_fold)]
        
        #--- Nested cv 
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            test_df = pd.concat(np.array(all_split_df)[folds_test])
        
            train_df, val_df, test_df = norm_data(train_df, val_df, test_df)
                
            dm.setup_kfold(train_df, val_df, test_df)
            
            #--- Init model
            model = self._get_model()
           
            #--- trainer main loop
            mf.pytorch.autolog()
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                x = dm.train_dataloader().dataset.all_ppg
                y = dm.train_dataloader().dataset.all_label

                model.fit(x,y)

                metrics = {}
                for mode in ['tr', 'val', 'ts']:
                    metrics.update(self.get_cv_metrics(fold_errors, dm, model, mode=mode))
                # logger.info(f"\t {metrics}")
                mf.log_metrics(metrics)

            #--- Save to model directory
            os.makedirs(self.config.path.model_directory, exist_ok=True)
            if self.config.exp.model_type=='mlp':
                joblib.dump(model, "{}/{}-fold{}-test_sp={:.3f}-test_dp={:.3f}.ckpt".format(
                                                                self.config.path.model_directory,
                                                                self.config.exp.exp_name,
                                                                foldIdx, 
                                                                metrics[f"ts/{target[0]}_mae"],
                                                                metrics[f"ts/{target[1]}_mae"]))
            else:
                joblib.dump(model, "{}/{}-fold{}-test_sp={:.3f}.ckpt".format(
                                                                self.config.path.model_directory,
                                                                self.config.exp.exp_name,
                                                                foldIdx, 
                                                                metrics[f"ts/{target[0]}_mae"]))
                
            
        #--- compute final metric
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items() if len(v)!=0 }
        err_dict = {tar:fold_errors[f"ts_{tar}_naive"] - fold_errors[f"ts_{tar}_label"] for tar in target}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)
        
        for mode in ['tr', 'val', 'ts']:
            err_dict = {tar:fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] for tar in target}
            tmp_metric = cal_metric(err_dict, mode=mode)
            out_metric.update(tmp_metric)            
        
        logger.info(f"\t {out_metric}")
        
        return out_metric

