from sklearn.linear_model import LogisticRegression
import numpy as np



class Trainer:
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_orig             = X_train
        self.X_train            = X_train
        
        self.X_train_dense_orig = X_train.toarray()
        self.X_train_dense      = self.X_train_dense_orig
        
        self.X_test        = X_test
        self.X_test_dense  = X_test.toarray()
        
        self.y_orig  = y_train.copy()
        self.y_train = y_train
        self.y_test  = y_test
        
        self.deltas = np.ones([X_train.shape[0]], dtype= np.bool)
        self.model  = LogisticRegression(C=1, solver = "lbfgs",
                                         max_iter = 800,
                                         fit_intercept= False,
                                         warm_start = True)
        
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        
        
    def remove(self, idx):
        self.deltas[idx] = 0
        self.X_train       = self.X_orig[self.deltas, :]
        self.y_train       = self.y_orig[self.deltas]
        self.X_train_dense = self.X_train_dense_orig[self.deltas, :]
        
    
    def correct_label(self, idx):
        self.y_orig[idx] = 1- self.y_orig[idx]
        self.y_train = self.y_orig[self.deltas]