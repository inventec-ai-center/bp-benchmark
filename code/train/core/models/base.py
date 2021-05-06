import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm
from core.utils import ContrastiveLoss, _get_siamese_input

def to_tensor(x):
    if not torch.is_tensor(x): x = torch.tensor(x)
    if torch.cuda.is_available(): x = x.cuda()
    return x

def to_numpy(x):
    if x.is_cuda: return x.detach().cpu().data.numpy()
    return x.detach().data.numpy()

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        
    def forward(self, x):
        if self.training:
            prob_mask = torch.FloatTensor(x.shape).uniform_(0,1).to(x.device) <= self.p
            return x * prob_mask
        return x * self.p

class Regressor:
    DEFAULTS = {}   
    _engine = None
    _optimizer = None
    def __init__(self, config, random_state):
        self.__dict__.update(self.DEFAULTS, **config)

        # Ensure reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        if "lambda_cal_l2" not in config:
            self.lambda_cal_l2 = 0.01
        
        self.use_scheduler = self.use_scheduler.lower() == "true"
        self.criteria = ContrastiveLoss()

    def fit(self, loader):
        if self.use_scheduler == True:
            print("Running experiment usign scheduler")
            trainer = torch.optim.lr_scheduler.OneCycleLR(self._optimizer, max_lr=self.lr, steps_per_epoch=len(loader), epochs=self.N_epoch)
        
        # if self.model_type=="mlp_siamese":
        #     for epoch in tqdm(range(self.N_epoch)):
        #         epoch_loss = []
        #         for itr, (x_data, x_cat, x_signal, y) in enumerate(loader):
        #             input1, input2, labels = _get_siamese_input([x_data, x_cat, x_signal, y])
        #             input1 = [to_tensor(data) for data in input1] # x_data, x_cat, x_signal, y = input1
        #             input2 = [to_tensor(data) for data in input2]
        #             y = to_tensor(labels)
                
        #             pred1, pred2 = self._engine.forward_twice(input1, input2) 
        #             assert pred1.shape == pred2.shape == y.shape
        #             total_loss = self.criteria(pred1, pred2, y)

        #             self._optimizer.zero_grad()
        #             total_loss.backward()
        #             self._optimizer.step()

        #             epoch_loss.append(total_loss.item())

        #             if self.use_scheduler:
        #                 trainer.step()

        #         # Log diagnostics
        #         if (epoch+1) % self.sample_step == 0:
        #             print("[{}/{}] Loss: {:.8f}".format(epoch+1, self.N_epoch, np.mean(epoch_loss)))        
        # else:
        for epoch in tqdm(range(self.N_epoch)):
            epoch_loss = []
            for itr, (x_data, x_cat, x_signal, y) in enumerate(loader):

                # Convert the input as tensor
                x_data = to_tensor(x_data)
                x_cat = to_tensor(x_cat)
                x_signal = to_tensor(x_signal)
                y = to_tensor(y)

                pred = self._engine(x_data, x_cat, x_signal)                                      
                assert pred.shape == y.shape
                total_loss = torch.mean((pred - y)**2)

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                epoch_loss.append(total_loss.item())

                if self.use_scheduler:
                    trainer.step()

            # Log diagnostics
            if (epoch+1) % self.sample_step == 0:
                print("[{}/{}] Loss: {:.8f}".format(epoch+1, self.N_epoch, np.mean(epoch_loss)))        

    def calibrate(self, loader, keep_dropout=True):
        # Initialize the layers
        engine_ft = copy.deepcopy(self._engine)            
        for item in engine_ft._modules:
            # Skip non-main layers
            if "main" not in item: continue
                
            # Remove dropout
            if keep_dropout == False:
                if item == "main":
                    engine_ft.main = nn.Sequential(*[x for x in engine_ft._modules[item] if isinstance(x, Dropout) == False])
                if item == "main_mlp":
                    engine_ft.main_mlp = nn.Sequential(*[x for x in engine_ft.main_mlp if isinstance(x, Dropout) == False])
                if item == "main_cnn":
                    engine_ft.main_cnn = nn.Sequential(*[x for x in engine_ft.main_cnn if isinstance(x, Dropout) == False])

            # Freeze layers
            for p in engine_ft._modules[item].parameters():
                p.requires_grad = False
                    
        # Initialize optimizer
        optimizer_ft = torch.optim.AdamW(engine_ft.parameters(), lr=self.lr_cal)                                
        engine_ft.train()
    
        # Run the calibration epochs                
        for epoch in range(self.N_epoch_calibration):
            for (x_data, x_cat, x_signal, y) in loader:
                # Convert the input as tensor
                x_data = to_tensor(x_data)
                x_cat = to_tensor(x_cat)
                x_signal = to_tensor(x_signal)
                y = to_tensor(y)
                
                # Compute loss
                pred = engine_ft(x_data, x_cat, x_signal)  
                assert pred.shape == y.shape
                loss_regression = torch.mean((pred - y)**2)

                # Regularization
                loss_l2 = 0
                for p1, p2 in zip(engine_ft.clf.parameters(), self._engine.clf.parameters()):
                    loss_l2 = torch.mean((p1-p2)**2)
              
                total_loss = loss_regression + self.lambda_cal_l2 * loss_l2
                
                optimizer_ft.zero_grad()
                total_loss.backward()
                optimizer_ft.step()
                            
        for item in engine_ft._modules:
            # Skip non-main layers
            if "main" not in item: continue
                
            # Freeze layers
            for p in engine_ft._modules[item].parameters():
                p.requires_grad = True
                
        self._engine = copy.deepcopy(engine_ft)
    
    def predict(self, loader, return_label=False):
        # Output buffer
        out_pred = []
        out_label = []

        # Inference
        with torch.no_grad():
            self._engine.eval()    
            for (x_data, x_cat, x_signal, y) in loader:

                x_data = to_tensor(x_data)
                x_cat = to_tensor(x_cat)
                x_signal = to_tensor(x_signal)

                y = to_tensor(y)
                pred = self._engine(x_data, x_cat, x_signal)

                # Make sure the prediction and the label has the same shape
                assert pred.shape == y.shape
                
                out_pred.append(to_numpy(pred))
                out_label.append(to_numpy(y))

            self._engine.train()

        # Stacking
        out_pred = np.concatenate(out_pred, axis=0)
        out_label = np.concatenate(out_label, axis=0)

        if return_label: return out_pred, out_label
        
    def save(self, model_path):
        save_dict = {
            "model_state_dict": self._engine.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict()
        }
        torch.save(save_dict, model_path)
        
    def load(self, model_path):
        if torch.cuda.is_available() == False: device = torch.device('cpu')
        else: device = torch.device("cuda:0")

        checkpoint = torch.load(model_path)
        self._engine.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
