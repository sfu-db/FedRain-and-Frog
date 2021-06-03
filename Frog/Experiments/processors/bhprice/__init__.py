from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from mlsql import ModelManager
from mlsql.processor import ComplaintRet, Processor
from mlsql import minimal_set_count_fix
from mlsql.utils import safe_transform
from sklearn.datasets import load_boston
from sklearn import preprocessing

# 6000 vs 25000
class BostonHousingPriceProcessor(Processor):
    Xtrain: tf.Tensor
    ytrain: tf.Tensor

    def __init__(self, seed: int):
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
        super().__init__(seed=seed)
        
        X, y = load_boston(return_X_y=True)
        X = preprocessing.normalize(X, norm='l2') 
        y = [(1 if i >= 21 else 0) for i in y]
        
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.2, random_state=self.auto_seed,
        )
        

        self.set_tensor_variables(
            Xtrain=Xtrain,
            ytrain=ytrain,
            Xtest=Xtest,
            ytest=ytest,  # Not one-hot encoded
        )

        