import os
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
# from .base_pl import Regressor

def toy(kernel, C):
    regressor = SVR(kernel=kernel, C=C)
    return regressor

class ToyModel:
    def __init__(self, param_model, random_state=100):
        super(ToyModel,self).__init__()
        self.model = toy(param_model.kernel, param_model.C)
    
    def fit(self, x, y):
        self.model.fit(x,y)

    def evaluate(self, x):
        pred = self.model.predict(x)
        return pred
