from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from numpy import genfromtxt

from ..processor import ComplaintRet, Processor

# 6000 vs 25000
class DiabetesCorrProcessor(Processor):
    def __init__(self):
        super().__init__()
#         x_train = genfromtxt('../processors/corruption_preprocess/diabetes_data/x_train.csv', delimiter=',')
#         y_train = genfromtxt('../processors/corruption_preprocess/diabetes_data/y_train.csv', delimiter=',')
#         x_test = genfromtxt('../processors/corruption_preprocess/diabetes_data/x_test.csv', delimiter=',')
#         y_test = genfromtxt('../processors/corruption_preprocess/diabetes_data/y_test.csv', delimiter=',')
#         x_query = genfromtxt('../processors/corruption_preprocess/diabetes_data/x_query.csv', delimiter=',')
#         y_query = genfromtxt('../processors/corruption_preprocess/diabetes_data/y_query.csv', delimiter=',')
#         y_corr = genfromtxt('../processors/corruption_preprocess/diabetes_data/y_corr.csv', delimiter=',')
#         corrsel = genfromtxt('../processors/corruption_preprocess/diabetes_data/corrsel.csv', delimiter=',')
        
        x, y = datasets.load_diabetes(return_X_y=True)
        x = preprocessing.normalize(x, norm='l2') 
        y = [(1 if i >= 140.5 else 0) for i in y]
        
        random_state = int(tf.random.uniform([], maxval=2**10, seed=1))
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=random_state,
        )
        x_test, x_query, y_test, y_query = train_test_split(
            x_test, y_test, test_size=0.5, random_state=random_state,
        )
        
        y_clean = y_train

#         # Extract numpy arrays for labels
#         ytrain = tf.one_hot(ytrain.astype(int), 2)
#         ytest = tf.one_hot(ytest.astype(int), 2)
        a_features = 5
        x_a_train = x_train[:, 0:a_features]
        x_b_train = x_train[:, a_features:]
        x_a_test = x_test[:, 0:a_features]
        x_b_test = x_test[:, a_features:]
        x_a_query = x_query[:, 0:a_features]
        x_b_query = x_query[:, a_features:]
        
        y_corr=y_train.copy()
        candidates = np.where(np.array(y_corr) == 1)[0]
        corruptions = int(len(candidates) * 0.3)
        np.random.seed(1024)
        corrupt_idx = np.random.choice(candidates, size=corruptions, replace=False)
        corrsel = np.full((len(y),), False)
        corrsel[corrupt_idx] = True
        for j in corrupt_idx:
            y_corr[j] = 0
        
        self.set_tensor_variables(
            x_train=x_train,
            y_train=y_train,
            x_a_train=x_a_train,
            x_b_train=x_b_train,
            x_test=x_test,
            y_test=y_test,
            x_a_test=x_a_test,
            x_b_test=x_b_test,
            x_query=x_query,
            y_query=y_query,
            x_a_query=x_a_query,
            x_b_query=x_b_query,
            y_corr=y_corr,
            corrsel=corrsel,
        )


    def complain(self, manager, exact=False):
        predict = manager.model(self.x_query)
        AQ = tf.reduce_sum(predict)
        AC = float(len(np.where(np.array(self.y_query) == 1)[0]))
        return ComplaintRet(AC=tf.stack([AC]), AQ=tf.stack([AQ])) 

    def test_complain(self, manager, exact=False):
        predict = manager.model(self.x_a_query, self.x_b_query)
        AQ = tf.reduce_sum(predict)
        AC = float(len(np.where(np.array(self.y_query) == 1)[0]))
        return ComplaintRet(AC=tf.stack([AC]), AQ=tf.stack([AQ])) 

    def lc_complain(self, manager_a, manager_b):
        predict = manager_a.model(self.x_a_query) + manager_b.model(self.x_b_query)
        AQ = tf.reduce_sum(predict)
        AC = float(len(np.where(np.array(self.y_query) == 1)[0]))
        return ComplaintRet(AC=tf.stack([AC]), AQ=tf.stack([AQ]))
    
    def query_data(self):
        return self.x_a_query, self.x_b_query    
    
    def fl_complain(self):
        """
        Complaint: Male ratio - Female ratio = 0
        Return: AC/PC: expected query output, no need to explain
                ids: index;
                agg: aggregation method (choices ["sum", "avg", "count?"])
                groupby: # of groupby categories, indicating IDs is "list" or "list of list"
                groupby_agg: groupby aggregation method (choice ["diff", "avg", "sum"])
        """
        AC = float(len(np.where(np.array(self.y_query) == 1)[0]))
        agg = "sum"
        idx = np.array([[1.0]] * len(self.y_query))
        return ComplaintRet(AC=AC, Ids=idx, Agg=agg, Groupby=None, Groupbyagg=None) 
    
