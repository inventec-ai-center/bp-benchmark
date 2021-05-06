import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Regressor

class LinearRegressionArch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionArch, self).__init__()
        self.clf = nn.Linear(input_size, output_size)

    def forward(self, x_tab, x_cat, x_signal):
        # Use only the tabular data
        x = x_tab

        # Rank for X should be 2: (N, F)
        if len(x.shape) !=2:
            raise RuntimeError("Tensor rank ({}) is not supported!".format(len(x.shape)))
        return self.clf(x)
                
class LinearRegression(Regressor):
    def __init__(self, config, random_state):
        super(LinearRegression, self).__init__(config, random_state)

        self._engine = LinearRegressionArch(self.input_size, self.output_size)
        self._optimizer = torch.optim.AdamW(self._engine.parameters(), lr=self.lr)  # Optimizer

        if torch.cuda.is_available():
            self._engine.cuda()

