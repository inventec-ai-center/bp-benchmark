from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

class adaModel:
    def __init__(self, param_model, random_state=100):
        super(adaModel,self).__init__()
        if param_model['max_depth']=='None':    param_model['max_depth'] = None
        dt = DecisionTreeRegressor(max_depth=param_model['max_depth'],
                                   min_samples_leaf=param_model['min_samples'], 
                                   random_state=random_state)
        self.model = AdaBoostRegressor(dt,
                                       n_estimators=param_model['n_estimators'],
                                       random_state=random_state)
    def fit(self, x, y):
        self.model.fit(x,y)

    def evaluate(self, x):
        pred = self.model.predict(x)
        return pred