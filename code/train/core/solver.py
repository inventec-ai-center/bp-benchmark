import json
import sys
import os 
PATHS = json.load(open("./paths.json"))
for k in PATHS: 
    sys.path.append(PATHS[k])
import mlflow as mf
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import os
import pprint
import hashlib
import copy

# Load models
from core.models.mlp import MLPVanilla, MLPCategorical, MLPCNN, MLPDilatedCNN
from core.models.cnn1d import CNNVanilla
# -- END --

# Load loaders
import copy
from core.loaders.passive_mimicv1 import PassiveMIMICv1

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# Load utilies
from core.utils import global_denorm, str2bool

class Solver:
    DEFAULTS = {}   
    def __init__(self, config):
        self.__dict__.update(self.DEFAULTS, **config)
        self.model_cache_path = "./model_cache/"
        self.config = config
        self.DATAROOT = self.dataset["data_path"] + self.loader_params["db_path"]
        self._load_mimicv1_csv()
        
        # Set default values for legacy experiments
        if "keep_dropout" not in config:
            self.keep_dropout = True
            
        self.keep_dropout = str2bool(self.keep_dropout)
        
    def _mae(self, pred, label, scales=None):
        assert pred.shape == label.shape
        denorm_stat = self.loader_params["target"].split(",")[0]

        if scales is None:
            pred = global_denorm(pred, denorm_stat)
            label = global_denorm(label, denorm_stat)
        else:
            min_stat, max_stat, mean_stat, std_stat = scales[denorm_stat]
            pred = pred * max_stat + min_stat
            label = label * max_stat + min_stat
            
        return np.mean(np.abs(pred - label))

    def _get_model(self):
        model = None
        # if self.model_type == "linear_regression":
        #     model = LinearRegression(self.model_params, self.random_state)
        if self.model_type == "mlp_vanilla":
            model = MLPVanilla(self.model_params, self.random_state)
        elif self.model_type == "mlp_categorical":
            model = MLPCategorical(self.model_params, self.random_state)
        elif self.model_type == "cnn_vanilla":
            model = CNNVanilla(self.model_params, self.random_state)
        elif self.model_type == "mlpcnn":
            model = MLPCNN(self.model_params, self.random_state)
        elif self.model_type == "mlpdilatedcnn":
            model = MLPDilatedCNN(self.model_params, self.random_state)
        # elif self.model_type == "mlp_siamese":
        #     model = MLPSiamese(self.model_params, self.random_state)
        else:
            raise RuntimeError("Model type {} is not supported!".format(self.model_type))
        return model

    def _load_mimicv1_csv(self):
        # Load the raw data 
        self._subjects = np.array(["484","225","437","216","417","284","438","471","213","439","237","240","446","281",
                                    "476","224","226","427","482","485","443","276","452","472","230"])
        self._xsubjectfold = {}
        mimicv1_meta = json.load(open(self.dataset + "../origin/mimic-1.0.0-meta.json"))

        anno = []
        for s in self._subjects:
            # Load the .csv
            df = pd.read_csv(self.DATAROOT + "{}.csv".format(s))  
            # Apply filtering over the rows, the following thresholds (obtained through eye-balling)
            mask = (df.sbp_std < 10) \
                    & ~(pd.isnull(df.ptt)) \
                    & (df.dbp_std < 5) \
                    & (df.ptt.between(100,600)) \
                    & (df.sbp_mean.between(60,200)) \
                    & (df.dbp_mean.between(30,160))
            df = df[mask].reset_index(drop=True)
            
            
            # TODO: Apply more strict PPG SQI
            if "ppg_sqi" in self.loader_params:
                df = df[df.ppg_sqi >= self.loader_params["ppg_sqi"]] 
            anno.append(df)
        anno = pd.concat(anno).reset_index(drop=True)
        anno["sbp"] = anno["sbp_mean"]
        anno["dbp"] = anno["dbp_mean"]
        anno["subject_id"] = anno["subject_id"].astype("str")
                
        # Create folds
        kfold = KFold(n_splits=self.N_fold)
        for s in self._subjects:
            df = anno[anno.subject_id == s]
            idxs = df.index.values

            for k, (train, test) in enumerate(kfold.split(idxs)):
                anno.loc[idxs[test],"IS_fold"] = k

        for k, (train, test) in enumerate(kfold.split(self._subjects)):
            self._xsubjectfold[k] = self._subjects[test]
            anno.loc[anno.subject_id.isin(self._subjects[test]), "XS_fold"] = k

        # Categorical data
        for s in mimicv1_meta:
            gender = 0 if mimicv1_meta[s]["gender"] == "m" else 1
            age = mimicv1_meta[s]["age"]

            # Assign values to the DataFrame
            anno.loc[anno.subject_id == s, "gender"] = gender
            anno.loc[anno.subject_id == s, "age"] = age
    
        anno["hod"] = pd.to_datetime(anno["rec_time"], format="%H-%M-%S").dt.hour
        
        print("Using {} number of rows".format(len(anno)))
        self.anno_data = anno

    def _get_gm_tag(self, split_type, train_folds, test_folds):
        config = copy.deepcopy(self.config)
        # Remove calibration specific configs
        config["model_params"].pop("N_cal_points")
        config["model_params"].pop("N_epoch_calibration")        
        config["model_params"].pop("lr_cal")
        try:
            config.pop("keep_dropout")
        except:
            pass
        
        config.pop("run_name")
        config.pop("root_dir")
        config.pop("model_dir")
        config.pop("log_dir")
        config.pop("sample_dir")

        # Include fold information
        config["split_type"] = split_type
        config["train_folds"] = train_folds
        config["test_folds"] = test_folds
        return hashlib.md5(str(config).encode()).hexdigest()

    def _get_loader(self, split_type, subjects, folds, mode):
        dataset = None
        if self.db_name == "mimicv1":
            dataset = PassiveMIMICv1(self.anno_data, 
                                        self.loader_params["features"].split(","), 
                                        self.loader_params["target"].split(","), 
                                        N_cal_points=self.model_params["N_cal_points"],

                                        # Runtime parameters
                                        split_type=split_type, 
                                        subjects=subjects,
                                        folds=folds, 
                                        mode=mode)

        else:
            raise RuntimeError("Database name {} is not supported!".format(self.db_path))

        # For small datasets, we can use higher batch_size for inference
        if mode == "test": batch_size=256 #batch_size = self.model_params["batch_size"]#len(dataset)
        elif mode == "cal": batch_size = 1
        else: batch_size = self.model_params["batch_size"]
        return  DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode=="train"))

    def evaluate_in_subject(self):
        '''
        Evaluating the generalization of general model to each of the known subjects.
        '''

        split_type="in-subject"
        testing_log = pd.DataFrame(columns=["fold","subject","N_cal_points","general_model_error","calibrated_model_error"])

        gm_errors = []
        cm_errors = []

        for k in range(self.N_fold):
            gm_fold_errors = []
            cm_fold_errors = []

            print("== IN-SUBJECT FOLD [{}/{}] ==".format(k+1, self.N_fold))
            train_folds = [x for x in np.arange(self.N_fold) if x != k]
            test_folds = [k]
        
            model = self._get_model()
            loader = self._get_loader(split_type, subjects=None, folds=train_folds, mode="train")

            print("-- Training --")
            gm_tag = self._get_gm_tag(split_type, train_folds, test_folds)
            cache_path = self.model_cache_path + "{}.pth".format(gm_tag)
            if os.path.exists(cache_path):
                print("Skip training general model, found {}".format(cache_path))
                model.load(cache_path)
            else:
                model.fit(loader)
                model.save(cache_path) 
                
            general_model = copy.deepcopy(model)

            print("-- Testing --")
            for subject in tqdm(self._subjects):            
                model = copy.deepcopy(general_model)  # Restore the parameters 
                cal_loader =  self._get_loader(split_type, subjects=[subject], folds=test_folds, mode="cal")
                test_loader =  self._get_loader(split_type, subjects=[subject], folds=test_folds, mode="test")

                # Compute performance for General Model
                pred, label = model.predict(test_loader, return_label=True)
                general_model_error = self._mae(pred, label)

                # Calibrate the general model
                model.calibrate(cal_loader, keep_dropout=self.keep_dropout)

                # Compute performance for the Calibrated Model
                pred, label = model.predict(test_loader, return_label=True)
                calibrated_model_error = self._mae(pred, label)
                
                # print("GM-Error: {} | CM-Error: {}".format(general_model_error, calibrated_model_error))
                testing_log = testing_log.append({"fold":k,
                                                    "subject":subject,
                                                    "N_cal_points":self.model_params["N_cal_points"],
                                                    "general_model_error":general_model_error,
                                                    "calibrated_model_error":calibrated_model_error},
                                                    ignore_index=True)

                gm_fold_errors.append(general_model_error)
                cm_fold_errors.append(calibrated_model_error)

            # Log the metrics
            gm_fold_errors = np.mean(gm_fold_errors)
            cm_fold_errors = np.mean(cm_fold_errors)
            gm_errors.append(gm_fold_errors)
            cm_errors.append(cm_fold_errors)
            mf.log_metric("IS_gm_valid_{}".format(k), gm_fold_errors)
            mf.log_metric("IS_cm_valid_{}".format(k), cm_fold_errors)
            print()

        mf.log_metric("IS_gm_cvmae", np.mean(gm_errors))
        mf.log_metric("IS_cm_cvmae", np.mean(cm_errors))

        print("--- DONE ---")
        print("Saving benchmark result of", self.exp_name, self.run_name)
        print("General Model AVGMAE: ", np.mean(gm_errors))
        print("Calibrated Model AVGMAE: ", np.mean(cm_errors))
        
        testing_log["exp_name"] = self.exp_name
        testing_log["run_name"] = self.run_name
        testing_log["model_type"] = self.model_type
        testing_log["db_name"] = self.db_name
        testing_log["features"] = self.loader_params["features"]
        testing_log["target"] = self.loader_params["target"]
        testing_log.to_csv(self.log_dir+"in-subject-log.csv", index=False)

    def evaluate_cross_subject(self):
        '''
        Evaluate the model on unknown subjects before.
        '''
        split_type = "cross-subject"
        testing_log = pd.DataFrame(columns=["fold","subject","N_cal_points","general_model_error","calibrated_model_error"])

        gm_errors = []
        cm_errors = []

        for k in range(self.N_fold):
            gm_fold_errors = []
            cm_fold_errors = []

            print("== CROSS-SUBJECT FOLD [{}/{}] ==".format(k+1, self.N_fold))
            train_folds = [x for x in np.arange(self.N_fold) if x != k]
            test_folds = [k]

            model = self._get_model()
            loader = self._get_loader(split_type=split_type, subjects=None, folds=train_folds, mode="train")

            print("-- Training --")

            # 1. Train General Model
            gm_tag = self._get_gm_tag(split_type, train_folds, test_folds)
            cache_path = self.model_cache_path + "{}.pth".format(gm_tag)
            
            if os.path.exists(cache_path):
                print("Skip training general model, found {}".format(cache_path))
                model.load(cache_path)
            else:
                model.fit(loader)
                model.save(cache_path) 
                
            general_model = copy.deepcopy(model)

            print("-- Testing --")
            # 2. Iterate for each subject
            test_subjects = self._xsubjectfold[k]
            for subject in tqdm(test_subjects):
                model = copy.deepcopy(general_model)  # Restore the parameters 
                cal_loader =  self._get_loader(split_type, subjects=[subject], folds=test_folds, mode="cal")
                test_loader =  self._get_loader(split_type, subjects=[subject], folds=test_folds, mode="test")
                                
                # Compute performance for General Model
                pred, label = model.predict(test_loader, return_label=True)
                general_model_error = self._mae(pred, label)
                # Calibrate the general model
                model.calibrate(cal_loader, keep_dropout=self.keep_dropout)

                # Compute performance for the Calibrated Model
                pred, label = model.predict(test_loader, return_label=True)
                calibrated_model_error = self._mae(pred, label)

                # Log the performance to the .csv
                testing_log = testing_log.append({"fold":k,
                                                    "subject":subject,
                                                    "N_cal_points":self.model_params["N_cal_points"],
                                                    "general_model_error":general_model_error,
                                                    "calibrated_model_error":calibrated_model_error},
                                                    ignore_index=True)

                gm_fold_errors.append(general_model_error)
                cm_fold_errors.append(calibrated_model_error)

            gm_fold_errors = np.mean(gm_fold_errors)
            cm_fold_errors = np.mean(cm_fold_errors)
            gm_errors.append(gm_fold_errors)
            cm_errors.append(cm_fold_errors)
            mf.log_metric("XS_gm_valid_{}".format(k), gm_fold_errors)
            mf.log_metric("XS_cm_valid_{}".format(k), cm_fold_errors)
            print()

        mf.log_metric("XS_gm_cvmae", np.mean(gm_errors))
        mf.log_metric("XS_cm_cvmae", np.mean(cm_errors))

        print("Saving benchmark result of", self.exp_name, self.run_name)        
        print("General Model AVGMAE: ", np.mean(gm_errors))
        print("Calibrated Model AVGMAE: ", np.mean(cm_errors))

        testing_log["exp_name"] = self.exp_name
        testing_log["run_name"] = self.run_name
        testing_log["model_type"] = self.model_type
        testing_log["db_name"] = self.db_name
        testing_log["features"] = self.loader_params["features"]
        testing_log["target"] = self.loader_params["target"]
        testing_log.to_csv(self.log_dir+"cross-subject-log.csv", index=False)
