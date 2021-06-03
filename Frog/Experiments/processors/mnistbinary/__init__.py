from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from ..processor import ComplaintRet, Processor
import mnist

# 6000 vs 25000
class MnistBinaryProcessor(Processor):
    Xtrain: tf.Tensor
    ytrain: tf.Tensor

    def __init__(self):
        """
        Parameters
        ----------
        seed : int
        corruption_rate : float
        complaint_mode : str
            Candidates: ["gender", "age", "both"]
        """

 
        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__()

        cwd = Path(__file__).resolve().parent
        
        train_images = mnist.train_images()
        train_images = train_images.reshape([train_images.shape[0], -1])/256
        train_labels = mnist.train_labels()%2

        test_images = mnist.test_images()
        test_images = test_images.reshape([test_images.shape[0], -1])/256
        test_labels = mnist.test_labels()%2
        
        random_state = int(tf.random.uniform([], maxval=2**10, seed=1))
        x_train, x_query, y_train, y_query = train_test_split(
            train_images, train_labels, test_size=0.1, random_state=random_state,
        )
        
        x_test, y_test = test_images, test_labels
        
        y_clean = y_train
        
        a_features = 300
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
        corrsel = np.full((len(y_train),), False)
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
       
        