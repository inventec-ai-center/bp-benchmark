import os 
from pathlib import Path
import sys
print(sys.path)
import json
import argparse
import torch
import torch.nn
import random
import numpy as np
import mlflow as mf
from core.signal_processing.utils import global_denorm
PATHS = json.load(open("../../paths.json"))
    
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def init_mlflow(exp_name, run_name):
    mf.set_tracking_uri(PATHS["PROJECT_PATH"] + "mlruns/")
    mf.set_experiment(exp_name)
    mf.start_run(run_name=run_name)
    
def init_dirs(config):
    active_run = mf.active_run()
    config["root_dir"] = PATHS["PROJECT_PATH"] + "/mlruns/{}/{}/artifacts/".format(active_run.info.experiment_id, active_run.info.run_id)
    for item in ["model_dir","sample_dir","log_dir"]:
        config[item] = config["root_dir"] + config[item]
        Path(config[item]).mkdir(parents=True, exist_ok=True)
    return config

def set_device(gpu_id):
    # Manage GPU availability
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if gpu_id != "": 
        torch.cuda.set_device(0)
        
    else:
        n_threads = torch.get_num_threads()
        n_threads = min(n_threads, 8)
        torch.set_num_threads(n_threads)
        print("Using {} CPU Core".format(n_threads))

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        y = torch.squeeze(y)
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

def _get_siamese_input(batch_data):
    batch_xdata, batch_xcat, batch_xsig, batch_y = batch_data
    # map sbp values to categories
    all_cat = np.array([_categorized(y) for y in batch_y])
    # generate category dictionary
    cat_dict = {i: np.where(all_cat == i)[0] for i in set(all_cat)}
    
    # for every data point, generate 1 positive example and 1 negative
    input1, input2, labels = [], [], []
    batch_data = list(zip(batch_xdata, batch_xcat, batch_xsig, batch_y))
    for data in batch_data:
        x_data, x_cat, x_sig, y = data
        y_cat = _categorized(y)
        # positive case
        pos_idx = cat_dict[y_cat]
        random.seed(100)
        rand_idx = random.randint(0,len(pos_idx))
        input1.append(data)
        input2.append(batch_data[pos_idx[rand_idx]])
        labels.append(0)
        # negative case
        neg_idx = [i for i in range(len(batch_data)) if i not in cat_dict[y_cat]]
        random.seed(100)
        rand_idx = random.randint(0,len(neg_idx))
        input1.append(data)
        input2.append(batch_data[neg_idx[rand_idx]])
        labels.append(1)
    # rearrange the data structure
    x_data = np.vstack([data[0] for data in input1])
    x_cat = np.vstack([data[1] for data in input1])
    x_sig = np.vstack([data[2] for data in input1])
    y = np.hstack([data[3] for data in input1])
    input1 = [x_data, x_cat, x_sig, y]
    
    x_data = np.vstack([data[0] for data in input2])
    x_cat = np.vstack([data[1] for data in input2])
    x_sig = np.vstack([data[2] for data in input2])
    y = np.hstack([data[3] for data in input2])
    input2 = [x_data, x_cat, x_sig, y]
    
    labels = np.hstack(labels).reshape(-1,1)
    return input1, input2, labels
        

def _categorized(sbp_value):
    sbp_value = global_denorm(sbp_value, "sbp")
    if sbp_value<=84:   return 0
    elif (sbp_value>84) and (sbp_value<=108): return 1
    elif (sbp_value>108) and (sbp_value<=132): return 2
    elif (sbp_value>132) and (sbp_value<=156): return 3
    elif (sbp_value>156) and (sbp_value<=180): return 4
    elif (sbp_value>180) and (sbp_value<=204): return 5
    elif (sbp_value>204) and (sbp_value<=228): return 6
    elif (sbp_value>228) and (sbp_value<=252): return 7
    elif (sbp_value>252) and (sbp_value<=276): return 8
    else: return 9
