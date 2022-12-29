import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from .base_pl import Regressor
import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

class MLPBP(Regressor):
    def __init__(self, param_model, random_state=0):
        super(MLPBP, self).__init__(param_model, random_state)

        self.model = MLPMixer(param_model.in_channels, 
                               param_model.dim, 
                               param_model.num_classes, 
                               param_model.num_patch,
                               param_model.depth, 
                               param_model.token_dim, 
                               param_model.channel_dim, 
                               param_model.dropout)
        
    def _shared_step(self, batch):
        x_ppg, y, x_abp, peakmask, vlymask = batch
        pred = self.model(x_ppg['ppg'])
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
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, num_patch,
                 depth, token_dim, channel_dim, dropout=0.2):
        super().__init__()

        self.num_patch = num_patch
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels, dim, kernel_size=1, stride=1),
            Rearrange('b c t -> b t c'),
        )

        # todo: LSTM emb
        self.lstm_patch_emb = nn.Sequential(
            Rearrange('b c t -> b t c'),
            nn.LSTM(input_size=in_channels, hidden_size=int(0.5*dim), num_layers=1,
                                      bidirectional=True, batch_first=True),
        )

        self.mixer_blocks = nn.ModuleList()
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.conv1d_decode = nn.Sequential(
            nn.Conv1d(num_patch, 2*num_patch, kernel_size=6, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv1d(2*num_patch, 4*num_patch, kernel_size=6, stride=4, padding=1),
            nn.ReLU(),
            # nn.Conv1d(512, 512, kernel_size=6, stride=4, padding=1)
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(4*num_patch, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x = self.to_patch_embedding(x)
        x, (hn, cn) = self.lstm_patch_emb(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.conv1d_decode(x)
        x = x.mean(dim=2)
        return self.mlp_head(x)