import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.base import Regressor, Dropout


class CNNVanillaArch(nn.Module):
    '''
    For every model architecture, we need to separate the classifier/regressor from the feature extraction layers. 
    Doing so allow us to easily isolate the final layers during the fine-tuning stage
    '''

    def __init__(self, input_size, output_size, hidden_size, dropout_prob):
        super(CNNVanillaArch, self).__init__()

        layers = []
        # in_channels, out_channels, kernel_size, stride = 1, padding = 0,
        layers.append(nn.Conv1d(input_size, hidden_size,
                                kernel_size=7, stride=5, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Conv1d(hidden_size, hidden_size//2,
                                kernel_size=7, stride=3, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(hidden_size//2))
        layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Conv1d(hidden_size//2, hidden_size //
                                4, kernel_size=7, stride=3, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(hidden_size//4))
        layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Conv1d(hidden_size//4, 8,
                                kernel_size=7, stride=3, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(8))
        layers.append(nn.Dropout(dropout_prob))
        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(208, 32))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(32, output_size))
        self.clf = nn.Sequential(*layers)

    def forward(self, x_tab, x_cat, x_signal):
        # Only use the signal input
        x = x_signal
        
        assert len(x.shape) == 3

        y = self.main(x)
        y = y.view(x.shape[0], -1)
        return self.clf(y)


class CNNVanilla(Regressor):
    def __init__(self, config, random_state):
        super(CNNVanilla, self).__init__(config, random_state)

        self._engine = CNNVanillaArch(self.input_size, self.output_size, self.hidden_size, self.dropout_prob)
        self._optimizer = torch.optim.AdamW(self._engine.parameters(), lr=self.lr)  # Optimizer
        
        if torch.cuda.is_available():
            self._engine.cuda()
