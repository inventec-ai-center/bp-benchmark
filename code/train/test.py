#%%
import os
import argparse

from time import time, ctime
from omegaconf import OmegaConf
from core.solver import Solver as solver_w2w
from core.solver_w2l import SolverW2l as solver_w2l

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

    solver.test()
    time_now = time()
    logger.warning(f"Time Used: {ctime(time_now-time_start)}")

    # =============================================================================
    # output
    # =============================================================================


#%%
if __name__=='__main__':
    parser = get_parser()
    main(parser.parse_args())