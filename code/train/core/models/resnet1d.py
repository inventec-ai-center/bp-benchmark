#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_pl import Regressor
from .resnet import MyConv1dPadSame, MyMaxPool1dPadSame, BasicBlock
import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

class Resnet1d(Regressor):
    def __init__(self, param_model, random_state=0):
        super(Resnet1d, self).__init__(param_model, random_state)

        self.model = ResNet1D(param_model.in_channel, param_model.base_filters,
                                param_model.first_kernel_size, param_model.kernel_size, 
                                param_model.stride, param_model.groups, param_model.n_block,
                                param_model.output_size, param_model.is_se, param_model.se_ch_low)
    def _shared_step(self, batch):
        x_ppg, y, x_abp, peakmask, vlymask = batch
        pred = self.model(x_ppg)
        loss = self.criterion(pred, y)
        return loss, pred, x_abp, y

    def training_step(self, batch, batch_idx):
        loss, pred_bp, t_abp, label = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label}
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_bp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_bp"] for v in train_step_outputs], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        loss, pred_bp, t_abp, label = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in val_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        loss, pred_bp, t_abp, label = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label}

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in test_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")
        return test_step_end_out

#%%

class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, first_kernel_size, kernel_size, stride, 
                        groups, n_block, output_size, is_se=False, se_ch_low=4, downsample_gap=2, 
                        increasefilter_gap=2, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.is_se = is_se
        self.se_ch_low = se_ch_low

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.first_kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.first_block_maxpool = MyMaxPool1dPadSame(kernel_size=self.stride)
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block,
                is_se=self.is_se,
                se_ch_low=self.se_ch_low)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

        # Classifier
        self.main_clf = nn.Linear(out_channels, output_size)

    # def forward(self, x):
    def forward(self, x):
        x = x['ppg']
        assert len(x.shape) == 3

        # skip batch norm if batchsize<4:
        if x.shape[0]<4:    self.use_bn = False 

        # first conv
        if self.verbose:
            logger.info('input shape', x.shape)
        out = self.first_block_conv(x)
        if self.verbose:
            logger.info('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_block_maxpool(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                logger.info('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                logger.info(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        h = self.final_relu(out)
        h = h.mean(-1) # (n_batch, out_channels)
        # logger.info('final pooling', h.shape)

        # ===== Concat x_demo
        out = self.main_clf(h)
        return out

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.414)       

#%%
if __name__=='__main__':
    from omegaconf import OmegaConf
    import pandas as pd
    import numpy as np
    import joblib
    import os
    os.chdir('/sensorsbp/code/train')
    from core.loaders.wav_loader import WavDataModule
    from core.utils import get_nested_fold_idx, cal_statistics
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import LearningRateMonitor
    from core.models.trainer import MyTrainer

    config = OmegaConf.load('/sensorsbp/code/train/core/config/unet_sensors_12s.yaml')
    all_split_df = joblib.load(config.exp.subject_dict)
    config = cal_statistics(config, all_split_df)
    for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(5)):
        if foldIdx==0:  break
    train_df = pd.concat(np.array(all_split_df)[folds_train])
    val_df = pd.concat(np.array(all_split_df)[folds_val])
    test_df = pd.concat(np.array(all_split_df)[folds_test])

    dm = WavDataModule(config)
    dm.setup_kfold(train_df, val_df, test_df)
    # dm.train_dataloader()
    # dm.val_dataloader()
    # dm.test_dataloader()
    
    # init model
    model = Unet1d(config.param_model)
    early_stop_callback = EarlyStopping(**dict(config.param_early_stop))
    checkpoint_callback = ModelCheckpoint(**dict(config.logger.param_ckpt))
    lr_logger = LearningRateMonitor()
    
    trainer = MyTrainer(**dict(config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ])

    # trainer main loop
    trainer.fit(model, dm)