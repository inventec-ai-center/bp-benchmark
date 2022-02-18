import os
import argparse
import random
import numpy as np
import torch

# Load modules
from core.solver_s2s import Solver as solver_s2s
from core.solver_s2l import SolverS2l as solver_s2l
from core.solver_f2l import SolverF2l as solver_f2l
from core.utils import log_params_mlflow, init_mlflow

from omegaconf import OmegaConf
from time import time, ctime
import mlflow as mf
from shutil import rmtree
from pathlib import Path

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) 
torch.cuda.manual_seed_all(SEED)

def get_parser():
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument("--config_file", type=str, help="Path for the config file") 

    return parser


def main(args):        
    if os.path.exists(args.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args.config_file))

    time_start = time()
    config = OmegaConf.load(args.config_file)
    
    #--- get the solver
    if config.exp.model_type=='unet1d':
        solver = solver_s2s(config)
    elif config.exp.model_type=='resnet1d':
        solver = solver_s2l(config)
    else:
        solver = solver_f2l(config)
    
    #--- training and logging into mlflow
    init_mlflow(config)
    with mf.start_run(run_name=f"{config.exp.N_fold}fold_CV_Results") as run:
        log_params_mlflow(config)
        cv_metrics = solver.evaluate()
        logger.info(cv_metrics)
        mf.log_metrics(cv_metrics)
    time_now = time()
    logger.warning(f"Time Used: {ctime(time_now-time_start)}")

    #--- remove automatically generated folder, since models are saved in mlflow folder
    pytorch_lightning_ckpt_dir = Path("./lightning_logs/")
    if pytorch_lightning_ckpt_dir.exists(): rmtree(pytorch_lightning_ckpt_dir)



if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())

