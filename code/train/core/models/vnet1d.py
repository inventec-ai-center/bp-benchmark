#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_pl import Regressor


#%%
class Vnet1d(Regressor):
    def __init__(self, param_model, random_state=0):
        super(Vnet1d, self).__init__(param_model, random_state)
        
        
        self.n_channels = param_model.n_channels #1
        self.f_kernel_size = param_model.f_kernel_size #5
        self.f_out_ch = param_model.f_out_ch #16
        self.f_padding = param_model.f_padding #2

        self.num_convolutions = param_model.num_convolutions#(1,2,3)
        self.kernel_size = param_model.kernel_size #5
        self.factor = param_model.factor #2
        self.dropout_rate = param_model.dropout_rate #0.2
        self.bottom_convolutions = param_model.bottom_convolutions #3
        self.up_pad = param_model.up_pad #2 
        self.up_out_pad = param_model.up_out_pad#0
        
        self.verbose=False
        
        self.model = Vnet1dCore(self.n_channels, self.f_kernel_size, self.f_out_ch, self.f_padding,
                            self.num_convolutions, self.kernel_size, self.factor, self.dropout_rate, 
                            self.bottom_convolutions, self.up_pad, self.up_out_pad, self.verbose)
        

    def _shared_step(self, batch):
        x, y, x_abp, peakmask, vlymask = batch
        pred = self.model(x['ppg'])
        loss = self.criterion(pred, x_abp)
        return loss, pred, x_abp, y, peakmask, vlymask

    def training_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label, peakmask, vlymask = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss":loss, "pred_abp":pred_abp, "true_abp":t_abp, "true_bp":label, "mask_pk": peakmask, "mask_vly": vlymask}
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_abp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_abp"] for v in train_step_outputs], dim=0)
        mask_pk = torch.cat([v["mask_pk"] for v in train_step_outputs], dim=0)
        mask_vly = torch.cat([v["mask_vly"] for v in train_step_outputs], dim=0)

        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label, peakmask, vlymask = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss":loss, "pred_abp":pred_abp, "true_abp":t_abp, "true_bp":label, "mask_pk": peakmask, "mask_vly": vlymask}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_abp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_abp"] for v in val_step_end_out], dim=0)
        mask_pk = torch.cat([v["mask_pk"] for v in val_step_end_out], dim=0)
        mask_vly = torch.cat([v["mask_vly"] for v in val_step_end_out], dim=0)

        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label, peakmask, vlymask = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return {"loss":loss, "pred_abp":pred_abp, "true_abp":t_abp, "true_bp":label, "mask_pk": peakmask, "mask_vly": vlymask}

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_abp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_abp"] for v in test_step_end_out], dim=0)
        mask_pk = torch.cat([v["mask_pk"] for v in test_step_end_out], dim=0)
        mask_vly = torch.cat([v["mask_vly"] for v in test_step_end_out], dim=0)

        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")
        return test_step_end_out

    def _cal_metric(self, logit:torch.tensor, label:torch.tensor):
        mse = torch.mean((logit-label)**2)
        mae = torch.mean(torch.abs(logit-label))
        me = torch.mean(logit-label)
        std = torch.std(torch.mean(logit-label, dim=1))
        return {"mse":mse, "mae":mae, "std": std, "me": me}    

        

#%%
class Vnet1dCore(nn.Module):
    def __init__(self, n_channels, f_kernel_size, f_out_ch, f_padding,
                 num_convolutions, kernel_size, factor, dropout_rate, 
                 bottom_convolutions, up_pad, up_out_pad, verbose=False):
        super(Vnet1dCore, self).__init__()
        
        self.num_levels = len(num_convolutions)
        
        self.init_layer = nn.Sequential(*[
            nn.Conv1d(n_channels, f_out_ch, kernel_size=f_kernel_size, stride=1, padding=f_padding, padding_mode='replicate'),
            nn.BatchNorm1d(f_out_ch),
            nn.ReLU()])
        
        self.down_layers = nn.ModuleList()
        out_ch = f_out_ch
        for l in range(self.num_levels):
            self.down_layers.append(DownConvBlock(out_ch, kernel_size, num_convolutions[l], dropout_rate, factor, verbose))
            out_ch = factor*out_ch

        self.mid_layer = ConvBlock(out_ch, kernel_size, bottom_convolutions, dropout_rate, verbose)

        self.up_layers = nn.ModuleList()
        for l in reversed(range(self.num_levels)):
            self.up_layers.append(UpConvBlock(out_ch, kernel_size, num_convolutions[l], dropout_rate, factor, up_pad, up_out_pad, verbose))
            out_ch = out_ch//factor

        self.last_layer = nn.Conv1d(f_out_ch, 1, kernel_size=1, stride=1, padding='same', bias=True)


    def forward(self,x):
        x = self.init_layer(x)
        
        features = []
        for l in range(self.num_levels):
            x, x_feats = self.down_layers[l](x)
            features.append(x_feats)

        # mid layer
        x = self.mid_layer(x)

        # decompression
        for l, f in zip(range(self.num_levels),reversed(features)):
            x = self.up_layers[l](x, f)

        return self.last_layer(x)
        
class ConvBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, n_convs, dropout, verbose=True):
        super(ConvBlock, self).__init__()
        
        self.n_convs = n_convs
        self.verbose = verbose
        
        self.cnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.ac_layers = nn.ModuleList()
        self.do_layers = nn.ModuleList()
        for i in range(n_convs):
            self.cnn_layers.append(nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding='same'))
            self.bn_layers.append(nn.BatchNorm1d(n_channels))
            self.ac_layers.append(nn.ReLU())
            self.do_layers.append(nn.Dropout2d(p=dropout))
            
    def forward(self, x):
        x_ori = x
        for i in range(self.n_convs):
            x = self.cnn_layers[i](x)
            if i == self.n_convs - 1:
                x = x.add(x_ori)
            x = self.bn_layers[i](x)
            x = self.ac_layers[i](x)
            
            # https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
            x = x.permute(0, 2, 1)
            x = self.do_layers[i](x)
            x = x.permute(0, 2, 1)
            if self.verbose: 
                print(f'{i} CNN block')
                print(x.shape)
        return x
    
    
class DownConvBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, n_convs, dropout, factor, verbose=True):
        super(DownConvBlock, self).__init__()
        
        self.n_convs = n_convs
        self.verbose = verbose
        
        self.conv_block = ConvBlock(n_channels, kernel_size, n_convs, dropout, verbose)
        self.down_conv = nn.Sequential(*[
            nn.Conv1d(n_channels, n_channels*factor, kernel_size=kernel_size, stride=factor, padding = (kernel_size)//2),
            nn.BatchNorm1d(n_channels*factor),
            nn.ReLU()])
    
    def forward(self, x):
        
        x = self.conv_block(x)
        
        x_feats = x
        x = self.down_conv(x) 
        if self.verbose: 
            print(f'Down')
            print(x.shape)
    
        return x, x_feats
    
    
class ConvBlock2(nn.Module):
    def __init__(self, n_channels, kernel_size, n_convs, dropout, verbose=True):
        super(ConvBlock2, self).__init__()
        
        self.n_convs = n_convs
        self.verbose = verbose
        
        self.cnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.ac_layers = nn.ModuleList()
        self.do_layers = nn.ModuleList()
        in_channels = n_channels*2
        for i in range(n_convs):
            in_n_channels = n_channels*2 if i == 0 else n_channels
            self.cnn_layers.append(nn.Conv1d(in_n_channels, n_channels, kernel_size=kernel_size, padding='same'))
            self.bn_layers.append(nn.BatchNorm1d(n_channels))
            self.ac_layers.append(nn.ReLU())
            self.do_layers.append(nn.Dropout2d(p=dropout))
            
        self.bn_last = nn.BatchNorm1d(n_channels)
        if n_convs == 1:
            self.bn_extra = nn.BatchNorm1d(n_channels)
            
    def forward(self, layer_input, x_feats):
        
        x = torch.cat([layer_input,x_feats], axis=1)
        if self.verbose: 
                print(f'Concat')
                print(x.shape)
        
        if self.n_convs == 1:
            x = self.cnn_layers[0](x)
            x = self.bn_extra(x)
            layer_input = self.bn_last(layer_input)
            x = x.add(layer_input)
            x = self.bn_layers[0](x)
            
            x = self.ac_layers[0](x)
            
            # https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
            x = x.permute(0, 2, 1)
            x = self.do_layers[0](x)
            x = x.permute(0, 2, 1)
            
            if self.verbose: 
                print(f'{0} CNN block')
                print(x.shape)
            
            return x
        
        for i in range(self.n_convs):
            x = self.cnn_layers[i](x)
            if i == self.n_convs - 1:
                layer_input = self.bn_last(layer_input)
                x = x.add(layer_input)
            x = self.bn_layers[i](x)
            x = self.ac_layers[i](x)
            
            # https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
            x = x.permute(0, 2, 1)
            x = self.do_layers[i](x)
            x = x.permute(0, 2, 1)
            if self.verbose: 
                print(f'{i} CNN block')
                print(x.shape)
                
        return x
    
    
class UpConvBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, n_convs, dropout, factor, pad, out_pad, verbose=True):
        super(UpConvBlock, self).__init__()
        
        self.n_convs = n_convs
        self.verbose = verbose
        
        self.up_conv = nn.Sequential(*[
            nn.ConvTranspose1d(n_channels, n_channels//factor, kernel_size, stride=factor, padding=pad, output_padding=out_pad),
            nn.BatchNorm1d(n_channels//factor),
            nn.ReLU()])
        
        self.conv_block = ConvBlock2(n_channels//factor, kernel_size, n_convs, dropout, verbose)
            
    
    def forward(self, x, x_feats):
        x = self.up_conv(x)
        if self.verbose:
            print(f'Up')
            print(x.shape)
        x = self.conv_block(x, x_feats)
        return x
    
    
