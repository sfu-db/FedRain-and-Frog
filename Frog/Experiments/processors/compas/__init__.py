from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

# To transform labels to one-hot encoding we use
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
# from mlsql import ModelManager
# from mlsql.processor import ComplaintRet, Processor
# from mlsql import minimal_set_count_fix, minimal_single_count_fix_multi
# from mlsql.utils import safe_transform
from ..processor import ComplaintRet, Processor

cwd = Path(__file__).resolve().parent

# 6000 vs 25000
class CompasProcessor(Processor):
    corruption_rate: float
    complaint_mode: str
    def __init__(self, seed: int, corruption_rate: float, complaint_mode: str):
        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__(
            seed=seed, corruption_rate=corruption_rate, complaint_mode=complaint_mode
        )
        raw_data = csv.reader(open(f"{cwd}/data/compas_proxy_data.csv"))
        
        raw_process_data = []
        for d in raw_data:
            raw_process_data.append(d)
        raw_data = raw_process_data
        name = raw_data[0]
        raw_data = raw_data[1:]

        X = [r[1:] for r in raw_data]
        Y = [r[0] for r in raw_data]
        print(len(raw_data[0]))
        
        train_X=[]
        train_Y=[]
        rename_dict = {
            "white": 1,
            "non-white":0,
            "1": 1,
            "0": 0,
        }
        for i,d in enumerate(X):
            train_X.append([rename_dict[feature] for feature in d])

        for i,d in enumerate(Y):
            train_Y.append(rename_dict[d])

        print(len(train_X[0]))
        Xtrain, X_test, ytrain, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)
        
        X_test_white= []
        X_test_nonwhite = []
        for d in X_test:
            if d[0] == 1:
                X_test_white.append(d)
            else:
                X_test_nonwhite.append(d)
        #Extract numpy arrays for labels
        ytrain = tf.one_hot(np.asarray(ytrain),2)
        y_test = tf.one_hot(np.asarray(y_test),2)
        self.set_tensor_variables(
            Xtrain=Xtrain,
            ytrain=ytrain,
            X_test_full=X_test,
            y_test_full=y_test,  # Not one-hot encoded
            X_test_white = X_test_white,
            X_test_nonwhite = X_test_nonwhite,
            Xcorr=[],
        )
    
        
    def complain(self, manager: ModelManager, exact: bool = True) -> ComplaintRet:
        return complain_impl(self.X_test_white, self.X_test_nonwhite, self.complaint_mask, manager, exact)


    def tiresias(self, manager: ModelManager) -> Tuple[tf.Tensor, tf.Tensor]:

        predictions_value = manager.predict(self.X_test_full).numpy()

        # The count that needs to be enforced
        count_true = np.sum(self.y_test_full.numpy()[:, 1])

        # Fix the test predictions based on complaint
        predictions_fixed = minimal_single_count_fix_multi(
            predictions_value, 2, 1, count_true
        )

        # Detect the label differences between the fix and the predictions
        changed_mask = predictions_fixed != predictions_value
        changed_X = self.X_test_full[changed_mask]
        changed_y = predictions_fixed[changed_mask]
        changed_y = tf.constant(
            self.enc.transform(changed_y.reshape(-1, 1)), dtype=tf.float32, name="tsy"
        )

        return changed_X, changed_y

@tf.function
def complain_impl(
    X_test_white: tf.Tensor,
    X_test_nonwhite: tf.Tensor,
    complaint_mask: tf.Tensor,
    model: ModelManager,
    exact: bool = True,
) -> ComplaintRet:
    
    if exact:
        white_arrest = tf.one_hot(model.predict(X_test_white), 2)
        nonwhite_arrest = tf.one_hot(model.predict(X_test_nonwhite), 2)
#         age = tf.one_hot(model.predict(X_test_age), 2)
    else:
        white_arrest = model.predict_proba(X_test_white)
        nonwhite_arrest = model.predict_proba(X_test_nonwhite)
    # predicted results using test data
#     threshold = 0.1
    Q_white = tf.reduce_mean(white_arrest[:, 1])
    Q_nonwhite = tf.reduce_mean(nonwhite_arrest[:, 1])
    margin = Q_nonwhite - Q_white
    AC = 0.0
    return ComplaintRet(AC=tf.stack([AC]), AQ=tf.stack([margin]))
