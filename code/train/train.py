# import argparse
# from pathlib import Path
# from core.config.config import GeneralConfig
# from core.wound_core import WoundCore


import os
import json
import argparse

import numpy as np 
# Local modules
import sys
PATHS = json.load(open("./paths.json"))
for k in PATHS: 
    sys.path.append(PATHS[k])
from core.solver import Solver
from core.utils import init_dirs, init_mlflow, str2bool, set_device

import mlflow as mf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 



DEFAULT_CONF_PATH = "core/config/mlpvanilla-lr-ptt-allfeat_0134.json"


def get_parser():
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument("--config_file", type=str, help="Path for the config file", default=DEFAULT_CONF_PATH) 
    parser.add_argument("--db_path", default="../../processed_data/processed_origin/") 
    parser.add_argument('--gpu_id', type=str, default="")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    return parser


def add_bool_arg(parser, full_name, abbreviate_name, help='', default=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + full_name, '-' + abbreviate_name,
                       dest=full_name, action='store_true', help=help)
    group.add_argument('--no-' + full_name, '-no-' + abbreviate_name,   
                       dest=full_name, action='store_false', help=help)
    parser.set_defaults(**{full_name: default})

def main(args):        
    if os.path.exists(args.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args.config_file))
    # Load config file
    config = json.load(open(args.config_file))
    config["model_params"]["lr"] = args.lr
    config["model_params"]["batch_size"] = args.batch_size
    config["model_params"]["hidden_size"] = args.hidden_size

    if args.debug:
        config["exp_name"] = "DEBUGGING"
        print("Running {} on DEBUG MODE!".format(args.config_file))
    
    #Initialize the experiment
    init_mlflow(config["exp_name"], config["run_name"])
    config = init_dirs(config)  # Setup directories and modify directories
    set_device(args.gpu_id)

    # Store the config file and params
    json.dump(config, open(config["root_dir"]+"config.json","w"))
    tmp = config["loader_params"]["features"]

    if len(config["loader_params"]["features"]) > 240:
        print("Features too long, truncating the features for logging purposes")
        config["loader_params"]["features"] = config["loader_params"]["features"][:240]

    mf.log_params(config["model_params"])
    mf.log_params(config["loader_params"])

    config["loader_params"]["features"] = tmp
    
    # Run evaluations
    solver = Solver(config)
    solver.evaluate_cross_subject()

    # Set status to complete once everything passes
    mf.set_tag("status","complete")



if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())

