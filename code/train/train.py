# import argparse
# from pathlib import Path
# from core.config.config import GeneralConfig
# from core.wound_core import WoundCore


import os
import argparse
import numpy as np 

from core.solver import Solver as solver_w2w
from core.solver_w2l import Solver as solver_w2l
from core.utils import log_params_mlflow, init_mlflow
from omegaconf import OmegaConf
from time import time, ctime
import mlflow as mf
from shutil import rmtree
from pathlib import Path

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  



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
    if config.exp.model_type=='unet1d':
        solver = solver_w2w(config)
    elif config.exp.model_type=='resnet1d':
        solver = solver_w2l(config)
    init_mlflow(config)
    with mf.start_run(run_name=f"{config.exp.N_fold}fold_CV_Results") as run:
        log_params_mlflow(config)
        cv_metrics = solver.evaluate()
        print(cv_metrics)
        mf.log_metrics(cv_metrics)
    time_now = time()
    logger.warning(f"Time Used: {ctime(time_now-time_start)}")

    # =============================================================================
    # output
    # =============================================================================
    pytorch_lightning_ckpt_dir = Path("./lightning_logs/")
    if pytorch_lightning_ckpt_dir.exists(): rmtree(pytorch_lightning_ckpt_dir)





if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())

