#%%
import pytorch_lightning as pl
import torch
import torch.nn as nn
from .utils import CosineWarmupScheduler

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  


class Regressor(pl.LightningModule):
    def __init__(self, param_model, random_state=0, pos_weight=None):
        super().__init__()
        # log hyperparameters
        self.param_model = param_model
        self.save_hyperparameters()

        # loss function
        self.criterion = nn.MSELoss()
    # =============================================================================
    # train / val / test
    # =============================================================================
    def forward(self, x):
       x = self.model(x)
       return x

    def _shared_step(self, batch):
        x_ppg, y, x_abp, peakmask, vlymask = batch
        pred, hidden = self.model(x_ppg)
        loss = self.criterion(pred, x_abp)
        return loss, pred, x_abp, y

    def training_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss":loss, "pred_abp":pred_abp, "true_abp":t_abp, "true_bp":label}
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_abp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_abp"] for v in train_step_outputs], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss":loss, "pred_abp":pred_abp, "true_abp":t_abp, "true_bp":label}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_abp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_abp"] for v in val_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return {"loss":loss, "pred_abp":pred_abp, "true_abp":t_abp, "true_bp":label}

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_abp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_abp"] for v in test_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")
        return test_step_end_out

    def _cal_metric(self, logit:torch.tensor, label:torch.tensor):
        mse = torch.mean((logit-label)**2)
        mae = torch.mean(torch.abs(logit-label))
        me = torch.mean(logit-label)
        std = torch.std(torch.mean(logit-label, dim=1))
        return {"mse":mse, "mae":mae, "std": std, "me": me} 
    
    def _log_metric(self, metrics, mode):
        for k,v in metrics.items():
            self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True)
    # =============================================================================
    # optimizer
    # =============================================================================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.param_model.lr)
        if self.param_model.get("scheduler_WarmUp"):
            logger.info("!!!!!!!! is using warm up !!!!!!!!")
            self.lr_scheduler = {"scheduler":CosineWarmupScheduler(optimizer,**(self.param_model.scheduler_WarmUp)), "monitor":"val_loss"}
            return [optimizer], self.lr_scheduler
        return optimizer
