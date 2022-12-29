from lightgbm import LGBMRegressor

class lgbModel:
    def __init__(self, param_model, random_state=100):
        super(lgbModel,self).__init__()
        
        self.model = LGBMRegressor(n_estimators=param_model['n_estimators'],
                          max_depth=param_model['max_depth'],
                          learning_rate=param_model['lr'],
                          num_leaves=param_model['leaves'], 
                          min_child_samples = param_model['min_samples'],
                          subsample_freq=1,
                          subsample=param_model['subsample'],
                          colsample_bytree=param_model['colsample_bytree'],
                          n_jobs=param_model.n_workers,
                          random_state=random_state)
    
    def fit(self, x, y):
        self.model.fit(x,y)

    def evaluate(self, x):
        pred = self.model.predict(x)
        return pred
