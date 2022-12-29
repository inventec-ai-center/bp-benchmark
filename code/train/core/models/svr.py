from sklearn.svm import SVR

class svrModel:
    def __init__(self, param_model, random_state=100):
        super(svrModel,self).__init__()
        
        self.model = SVR(C=param_model['C'], 
                         kernel=param_model['kernel'], 
                         gamma=param_model['gamma'], 
                         epsilon=param_model['epsilon'], 
                         cache_size=7000)
    def fit(self, x, y):
        self.model.fit(x,y)

    def evaluate(self, x):
        pred = self.model.predict(x)
        return pred
