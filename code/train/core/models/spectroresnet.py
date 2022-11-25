#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torchaudio.transforms import Spectrogram
from torchaudio.functional import amplitude_to_DB

from .base_pl import Regressor

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

class SpectroResnet(Regressor):
    def __init__(self, param_model, random_state=0):
        super(SpectroResnet, self).__init__(param_model, random_state)
        
        self.model = raw_signals_deep_ResNet(in_channel=param_model.in_channel, 
                                             num_filters=param_model.num_filters,
                                             num_res_blocks=param_model.num_res_blocks,
                                             cnn_per_res=param_model.cnn_per_res,
                                             kernel_sizes=param_model.kernel_sizes,
                                             max_filters=param_model.max_filters,
                                             pool_size=param_model.pool_size,
                                             pool_stride_size=param_model.pool_stride_size,
                                             n_dft=param_model.n_dft,
                                             n_hop=param_model.n_hop,
                                             fmin=param_model.fmin,
                                             fmax=param_model.fmax,
                                             mlp_size=param_model.mlp_size,
                                             mid_hidden=param_model.mid_hidden,
                                             gru_hidden=param_model.gru_hidden,
                                             UseDerivative=param_model.UseDerivative,
                                             verbose=param_model.model_verbose)
                                             
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
class single_channel_resnet(nn.Module):
    def __init__(self, in_channel=1, num_filters=32, num_res_blocks=4, cnn_per_res=3,
              kernel_sizes=[8, 5, 3], max_filters=64, pool_size=3, pool_stride_size=2, verbose=False):
        super(single_channel_resnet, self).__init__()
        self.verbose = verbose

        self.pool_size = pool_size
        self.pool_stride_size = pool_stride_size
        self.bn1 = nn.BatchNorm1d(in_channel)

        self.layers = nn.ModuleList()
        in_planes, planes = in_channel, num_filters
        for i in range(num_res_blocks):
            self.layers.append(self._make_block(i, in_planes, planes, cnn_per_res, kernel_sizes))
            in_planes = planes
            planes *= 2
            if planes > max_filters:
                planes = max_filters


    def _make_block(self, i_layer, in_planes, planes, cnn_per_res, kernel_sizes):
        res_in_planes = in_planes
        layers = []
        residue_block = []
        out_block = []
        for i in range(cnn_per_res):
            layers.append(nn.Conv1d(in_planes, planes, kernel_size=kernel_sizes[i], padding="same"))
            layers.append(nn.BatchNorm1d(planes))
            if (i < cnn_per_res - 1):
                layers.append(nn.ReLU())

            in_planes = planes
            
        if i_layer == 0:
            residue_block.append(nn.BatchNorm1d(res_in_planes))
        else:
            residue_block.append(nn.Conv1d(res_in_planes, planes, kernel_size=1, padding="same"))
            residue_block.append(nn.BatchNorm1d(planes))
        out_block.append(nn.ReLU())
        out_block.append(nn.AvgPool1d(kernel_size=self.pool_size, stride=self.pool_stride_size))
        
        return nn.ModuleList([nn.Sequential(*layers), nn.Sequential(*residue_block), nn.Sequential(*out_block)])
    
    def forward(self, x):
        input = x.clone()
        residue = x
        x = self.bn1(x)
        for i, (layer, residue_layer, out_layer) in enumerate(self.layers):
            if self.verbose: print(i, x.shape, residue.shape)
            x = layer(x)
            residue = residue_layer(residue)
            x += residue
            x = out_layer(x)
            residue = x
        return input, x

class mid_spectrogram_LSTM_layer(nn.Module):
    def __init__(self, n_dft=64, n_hop=64, fmin=0.0, fmax=25, feat_dim=32, mlp_size=351, verbose=False):
        super(mid_spectrogram_LSTM_layer, self).__init__()
        self.verbose = verbose

        self.n_dft = n_dft
        self.n_hop = n_hop
        self.feat_dim = feat_dim
        self.stft = Spectrogram(n_fft=n_dft, hop_length=n_hop, center=False)  # return power instead of complex, don't need extra magnitude function
        
        self.mlp_size = mlp_size
        self.mlp = nn.Linear(mlp_size,self.feat_dim) # miss the kernel_regularizer, l2_lambda is not used
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(feat_dim)
    
    def forward(self, x):
        x = self.stft(x)
        if self.verbose: print(x.shape)
        x = amplitude_to_DB(x, 10, 1e-05, 0) #db_multiplier = log10(max(ref, amin)), where ref=1 and amin=1e-5
        if self.verbose: print(x.shape)
        x = x.reshape(x.shape[0], -1)
        if self.verbose: print(x.shape)
        #print('mlp_size_shape',x.shape[-1])
        #print('mlp_size',self.mlp_size)
        x = self.bn(self.relu(self.mlp(x)))
        return x

class raw_signals_deep_ResNet(nn.Module):
    def __init__(self, in_channel=1, num_filters=32, num_res_blocks=4, cnn_per_res=3,
               kernel_sizes=[8, 5, 3], max_filters=64, pool_size=3, pool_stride_size=2,
               n_dft=64, n_hop=64, fmin=0.0, fmax=25, mlp_size=351, mid_hidden=64, gru_hidden=64, UseDerivative=False, verbose=False):
        super(raw_signals_deep_ResNet, self).__init__()
        self.verbose = verbose

        self.UseDerivative = UseDerivative
        n_channel = 3 if self.UseDerivative else 1

        self.layers = nn.ModuleList()
        self.mid_spec_layers = nn.ModuleList()
        for _ in range(n_channel):
            self.layers.append(single_channel_resnet(in_channel=in_channel, num_filters=num_filters, 
                                  num_res_blocks=num_res_blocks, cnn_per_res=cnn_per_res, 
                                  kernel_sizes=kernel_sizes, max_filters=max_filters, 
                                  pool_size=pool_size, pool_stride_size=pool_stride_size, verbose=self.verbose))
            self.mid_spec_layers.append(mid_spectrogram_LSTM_layer(n_dft=n_dft, n_hop=n_hop, fmin=fmin, 
                                          fmax=fmax, feat_dim=mid_hidden, mlp_size=mlp_size, verbose=self.verbose))
        
        self.resnet_out_channel = num_filters * (2 ** (num_res_blocks - 1))
        self.resnet_out_channel = self.resnet_out_channel if self.resnet_out_channel <= max_filters else max_filters
        
        ## MODIFIED
        self.sig_bn1 = nn.BatchNorm1d(mid_hidden*n_channel)
        self.sig_gru = nn.GRU(mid_hidden*n_channel, gru_hidden, 1)
        self.sig_bn2 = nn.BatchNorm1d(gru_hidden)

        self.spec_bn = nn.BatchNorm1d(mid_hidden*n_channel)

        
        ## MODIFIED
        self.joint_mlp = nn.Sequential(*[
                            #nn.Linear(2 * self.resnet_out_channel, 32),
                            nn.Linear(gru_hidden+mid_hidden*n_channel, 32),
                            nn.ReLU(),
                            nn.Dropout(0.25),
                            nn.Linear(32, 32),
                            nn.ReLU(),
                            nn.Dropout(0.25)])

        self.reg_mlp = nn.Linear(32, 2)



    def forward(self, x):
        x_all = [x]
        if self.UseDerivative:
            x_dt1 = torch.diff(x, append=x[:,:,-1:])
            x_dt2 = torch.diff(x_dt1, append=x_dt1[:,:,-1:])
            x_all = [x, x_dt1, x_dt2]
        #print(x.shape, x_dt1.shape, x_dt2.shape)

        inputs = []
        channel_outputs = []
        for input, layer in zip(x_all, self.layers):
            channel_resnet_input, channel_resnet_out = layer(input)
            channel_outputs.append(channel_resnet_out)
            inputs.append(channel_resnet_input)
        
        spectral_outputs = []
        for input, layer in zip(inputs, self.mid_spec_layers):
            spectro_x = layer(input)
            spectral_outputs.append(spectro_x)

        spectral_outputs = torch.cat(spectral_outputs, dim=1)
        if self.verbose: print('spectral_outputs', spectral_outputs.shape)
        channel_outputs = torch.cat(channel_outputs, dim=1)
        if self.verbose: print('channel_outputs', channel_outputs.shape)
        
        # LETS DO OVERFIT
        x = self.sig_bn1(channel_outputs)
        if self.verbose: print('sig_bn1', x.shape)
        x = x.permute(0, 2, 1)
        x, _ = self.sig_gru(x)
        if self.verbose: print('sig_gru', x.shape)
        x = x[:,-1,:].squeeze(1) # take only the final time output
        x = self.sig_bn2(x)
        if self.verbose: print('sig_bn2', x.shape)

        # join time-domain and frequnecy domain fully-conencted layers
        s = self.spec_bn(spectral_outputs)
        if self.verbose: print('spec_bn', s.shape)
        out = torch.cat([x, s], dim=-1)
        if self.verbose: print('joint', out.shape)

        # LETS DO OVERFIT
        out = self.joint_mlp(out)
        if self.verbose: print('joint_mlp', out.shape)
        out = out.view(out.shape[0], -1)
        if self.verbose: print('flatten', out.shape)
        out = self.reg_mlp(out)
        if self.verbose: print('reg_mlp', out.shape)
        return out