from sklearn.neural_network import MLPRegressor

class mlpModel:
    def __init__(self, param_model, random_state=100):
        super(mlpModel,self).__init__()
        
        self.model = MLPRegressor(hidden_layer_sizes=param_model['hidden'],
                                  batch_size=64, 
                                  random_state=random_state)
    
    def fit(self, x, y):
        self.model.fit(x,y)

    def evaluate(self, x):
        pred = self.model.predict(x)
        return pred