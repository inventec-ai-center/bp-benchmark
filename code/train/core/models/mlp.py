import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.base import Regressor, Dropout

class MLPVanillaArch(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout_prob):
        super(MLPVanillaArch, self).__init__()
        # Feature extractor        

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(Dropout(dropout_prob))

        for i in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            layers.append(Dropout(dropout_prob))

        self.main = nn.Sequential(*layers)

        # Regressor
        self.clf = nn.Linear(hidden_size, output_size)

    def forward(self, x_tab, x_cat, x_signal):
        # Use only the tabular data
        x = x_tab

        # Rank for X should be 2: (N, F)
        if len(x.shape) !=2:
            raise RuntimeError("Tensor rank ({}) is not supported!".format(len(x.shape)))
            
        h = self.main(x)
        return self.clf(h)

class MLPVanilla(Regressor):
    def __init__(self, config, random_state):
        super(MLPVanilla, self).__init__(config, random_state)
        
        self._engine = MLPVanillaArch(self.input_size, self.output_size, self.hidden_size, 
                                        self.n_layers, self.dropout_prob)
        self._optimizer = torch.optim.AdamW(self._engine.parameters(), lr=self.lr)  # Optimizer
        if torch.cuda.is_available():
            self._engine.cuda()

class MLPCategoricalArch(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, embedding_sizes, n_layers, dropout_prob):
        super(MLPCategoricalArch, self).__init__()
        # THIS PART REQUIRES MANUAL MODIFICATION
        n_class, emb_size = embedding_sizes["hod"]
        self.emb_hod = nn.Embedding(n_class, emb_size)

        # Feature extractor
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(Dropout(dropout_prob))
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            layers.append(Dropout(dropout_prob))
        self.main = nn.Sequential(*layers)

        # Regressor
        self.clf = nn.Linear(hidden_size, output_size)

    def forward(self, x_tab, x_cat, x_signal):
        # Process categorical data
        emb_out = []
        emb_out.append(self.emb_hod(x_cat[:, [0]]).squeeze(1))  # Hour of Day
        emb_out.append(x_cat[:, [1]])  # Gender
        emb_out = torch.cat(emb_out, axis=1)

        # Concatenate categorical data with tabular data
        x = torch.cat([x_tab, emb_out], axis=1)

        # Feed-forward pass
        h = self.main(x)
        return self.clf(h)


class MLPCategorical(Regressor):
    def __init__(self, config, random_state):
        super(MLPCategorical, self).__init__(config, random_state)

        self._engine = MLPCategoricalArch(self.input_size, self.output_size, self.hidden_size, self.embedding_sizes,
                                        self.n_layers, self.dropout_prob)
        self._optimizer = torch.optim.AdamW(self._engine.parameters(), lr=self.lr)  # Optimizer
        if torch.cuda.is_available():
            self._engine.cuda()

class MLPCNNArch(nn.Module):
    '''
    For every model architecture, we need to separate the classifier/regressor from the feature extraction layers. 
    Doing so allow us to easily isolate the final layers during the fine-tuning stage
    '''
    def __init__(self, input_size_mlp, input_size_cnn, output_size, n_layers, hidden_size, dropout_prob, arch_type):
        super(MLPCNNArch, self).__init__()
        
        # Initialize extractors
        self._init_mlp(input_size_mlp + hidden_size//4, hidden_size, n_layers, dropout_prob)
        self._init_cnn(input_size_cnn, hidden_size, dropout_prob, arch_type)
        
        # Initialize regressor
        self.clf = nn.Linear(hidden_size, output_size)
        
    def _init_mlp(self, input_size, hidden_size, n_layers, dropout_prob):
        # Feature extractor
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.Dropout(dropout_prob))

        for i in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            layers.append(nn.Dropout(dropout_prob))

        self.main_mlp = nn.Sequential(*layers)
        
    def _init_cnn(self, input_size, hidden_size, dropout_prob, arch_type="wide"):
        layers = []
        # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
        
        if arch_type == "wide":
            layers.append(nn.Conv1d(input_size, hidden_size, kernel_size=21, stride=7, padding=0))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.Conv1d(hidden_size, hidden_size//4, kernel_size=16, stride=1, padding=0))
            
        else:
            layers.append(nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=3, padding=2))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            layers.append(nn.BatchNorm1d(hidden_size))
                        
            layers.append(nn.Conv1d(hidden_size, hidden_size//2, kernel_size=7, stride=3, padding=2))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            layers.append(nn.BatchNorm1d( hidden_size//2))
                        
            layers.append(nn.Conv1d(hidden_size//2, hidden_size//4, kernel_size=7, stride=3, padding=2))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            layers.append(nn.BatchNorm1d(hidden_size//4))
                        
            layers.append(nn.Conv1d(hidden_size//4, hidden_size//4, kernel_size=4, stride=1, padding=0))           
            pass

        self.main_cnn = nn.Sequential(*layers)
        
    def forward(self, x_tab, x_cat, x_signal):
        # Pass down the cleaned cycles through the CNN
        h_signal = self.main_cnn(x_signal).squeeze(-1)
        
        # Concatenate CNN's output with existing tabular data
        h = torch.cat([x_tab, h_signal], axis=-1)
        h = self.main_mlp(h)
        
        # Get the SBP through regressor
        return self.clf(h)
    
class MLPCNN(Regressor):
    def __init__(self, config, random_state):
        super(MLPCNN, self).__init__(config, random_state)
        
        # input_size_mlp, input_size_cnn, output_size, n_layers, hidden_size, dropout_prob
        self._engine = MLPCNNArch(self.input_size_mlp, self.input_size_cnn, self.output_size, self.n_layers, self.hidden_size, self.dropout_prob, self.arch_type)
        self._optimizer = torch.optim.AdamW(self._engine.parameters(), lr=self.lr)  # Optimizer
        if torch.cuda.is_available():
            self._engine.cuda()
            
            
class MLPDilatedCNNArch(nn.Module):
    '''
    For every model architecture, we need to separate the classifier/regressor from the feature extraction layers. 
    Doing so allow us to easily isolate the final layers during the fine-tuning stage
    '''
    def __init__(self, input_size_mlp, input_size_cnn, output_size, n_layers, hidden_size, dropout_prob):
        super(MLPDilatedCNNArch, self).__init__()
        
        # Initialize extractors
        self._init_mlp(input_size_mlp + hidden_size//4, hidden_size, n_layers, dropout_prob)
        self._init_cnn(input_size_cnn, hidden_size, dropout_prob)
        
        # Initialize regressor
        layers = []
        layers.append(nn.Linear(hidden_size, output_size))
        self.clf = nn.Sequential(*layers)
        
    def _init_mlp(self, input_size, hidden_size, n_layers, dropout_prob):
        # Feature extractor
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.Dropout(dropout_prob))

        for i in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            layers.append(nn.Dropout(dropout_prob))  

        self.main_mlp = nn.Sequential(*layers)
        
    def _init_cnn(self, input_size, hidden_size, dropout_prob):
        layers = []
        
        layers.append(nn.Conv1d(input_size, hidden_size, kernel_size=5, stride=2, padding=4, dilation=2))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(hidden_size))

        layers.append(nn.Conv1d(hidden_size, hidden_size//2, kernel_size=5, stride=2, padding=4, dilation=2))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(hidden_size//2))

        layers.append(nn.Conv1d(hidden_size//2, hidden_size//4, kernel_size=32, stride=1, padding=0))
        self.main_cnn = nn.Sequential(*layers)
        
    def forward(self, x_tab, x_cat, x_signal):
        # Pass down the cleaned cycles through the CNN
        h_signal = self.main_cnn(x_signal).squeeze(-1)
               
        # Concatenate CNN's output with existing tabular data
        h = torch.cat([x_tab, h_signal], axis=-1)
        h = self.main_mlp(h)
        
        # Get the SBP through regressor
        return self.clf(h)
    
    
class MLPDilatedCNN(Regressor):
    def __init__(self, config, random_state):
        super(MLPDilatedCNN, self).__init__(config, random_state)
        
        # input_size_mlp, input_size_cnn, output_size, n_layers, hidden_size, dropout_prob
        self._engine = MLPDilatedCNNArch(self.input_size_mlp, self.input_size_cnn, self.output_size, self.n_layers, self.hidden_size, self.dropout_prob)
        self._optimizer = torch.optim.AdamW(self._engine.parameters(), lr=self.lr)  # Optimizer
        if torch.cuda.is_available():
            self._engine.cuda()
