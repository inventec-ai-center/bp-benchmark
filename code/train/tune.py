#%%
from core.utils import log_params_mlflow, log_hydra_mlflow
from core.solver import Solver
from time import time, ctime

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

from shutil import rmtree
from pathlib import Path

import hydra
from hydra import utils
import omegaconf
import mlflow as mf
from mlflow.tracking.client import MlflowClient



#%%
@hydra.main(config_path='./core/config/hydra', config_name="unet_uci_5s")
def main(config):
    # =============================================================================
    # check config have been run
    # =============================================================================
    # generate a fake run to get the to be stored parameter format of this run
    MLRUNS_DIR = utils.get_original_cwd()+'/mlruns'
    client = MlflowClient(tracking_uri=MLRUNS_DIR)
    exp = client.get_experiment_by_name(config.exp.exp_name)
    if exp is not None:
        exp_id = exp.experiment_id

        fake_run_name = "fake_run"
        mf.set_tracking_uri(MLRUNS_DIR)
        mf.set_experiment(config.exp.exp_name)
        with mf.start_run(run_name=fake_run_name) as run:
            log_params_mlflow(config)
            fake_run_id = run.info.run_id
        fake_run = client.get_run(fake_run_id)
        fake_run_params = fake_run.data.params

        # delete all failed
        all_run_infos = client.list_run_infos(experiment_id=exp_id)
        for run_info in all_run_infos:
            if run_info.status=="FAILED":
                client.delete_run(run_info.run_id)

        # check all existed run
        all_run_infos = client.list_run_infos(experiment_id=exp_id)
        for run_info in all_run_infos:
            if run_info.run_id==fake_run_id: continue # skip fakerun itself
            run = client.get_run(run_info.run_id)
            params = run.data.params
            metrics = run.data.metrics
            if params==fake_run_params and run_info.status=="FINISHED":
                logger.warning(f'Find Exist Run with test/sbp_mae: {round(metrics["test/sbp_mae"],3)}')
                client.delete_run(fake_run_id)
                return metrics["test/sbp_mae"]
        client.delete_run(fake_run_id)


    # =============================================================================
    # initialize
    # =============================================================================
    # print(config)
    time_start = time()
    logger.info(f"Data Path: {Path(config.exp.subject_dict).absolute()} | MLflow Path: {Path(config.path.mlflow_dir).absolute()}")
    logger.info(f"Dataset:{config.exp.exp_name} | Model:{config.exp.model_type}")


    # =============================================================================
    # Setup Solver
    # =============================================================================
    solver = Solver(config)

#%%
    # =============================================================================
    # Run
    # =============================================================================
    mf.set_tracking_uri(MLRUNS_DIR)
    mf.set_experiment(config.exp.exp_name)
    with mf.start_run(run_name=f"{config.exp.N_fold}fold_CV_Results") as run:
        log_params_mlflow(config)
        cv_metrics = solver.evaluate()
        mf.log_metrics(cv_metrics)
        log_hydra_mlflow(name="tune")
        
    time_now = time()
    logger.warning(f"{config.exp.exp_name} Time Used: {ctime(time_now-time_start)}")
    return cv_metrics["test/sbp_mae"]


#%%
if __name__ == "__main__":
    main()
    # rmtree("./multirun")