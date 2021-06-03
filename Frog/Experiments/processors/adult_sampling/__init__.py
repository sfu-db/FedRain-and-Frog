from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from sklearn.preprocessing import OrdinalEncoder

from ..processor import ComplaintRet, Processor

# 6000 vs 25000
class SampledAdultProcessor(Processor):
    def __init__(self):
        super().__init__()       
        self.oe = OrdinalEncoder()
        df = pd.read_csv(f"processors/adultNoCorr/data/adult_proxy_data.csv")
        df = df.sample(2600)
        data = self.oe.fit_transform(df)
        np.random.shuffle(data)
        x = data[:, 1:]
        y = data[:, 0]
#         y[(y == 0)] = -1 # taylor

        random_state = int(tf.random.uniform([], maxval=2**10, seed=1))
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=random_state,
        )
        
        x_test, x_query, y_test, y_query = train_test_split(
            x_test, y_test, test_size=0.5, random_state=random_state,
        )

        x_test_male = x_test[x_test[:, 0] == 1]
        x_test_female = x_test[x_test[:, 0] == 0]
        
        x_query_male = x_query[x_query[:, 0] == 1]
        x_query_female = x_query[x_query[:, 0] == 0]

        # Extract numpy arrays for labels
#         ytrain = tf.one_hot(ytrain.astype(int), 2)
#         ytest = tf.one_hot(ytest.astype(int), 2)
        a_features = 9
        x_a_train = x_train[:, 0:a_features]
        x_b_train = x_train[:, a_features:]
        x_a_test = x_test[:, 0:a_features]
        x_b_test = x_test[:, a_features:]
        x_a_query = x_query[:, 0:a_features]
        x_b_query = x_query[:, a_features:]
        x_a_query_male = x_query_male[:, 0:a_features]
        x_b_query_male = x_query_male[:, a_features:]
        x_a_query_female = x_query_female[:, 0:a_features]
        x_b_query_female = x_query_female[:, a_features:]
        
        

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
            x_query_male=x_query_male,
            x_query_female=x_query_female,
            x_a_query=x_a_query,
            x_b_query=x_b_query,
            x_a_query_male=x_a_query_male,
            x_b_query_male=x_b_query_male,
            x_a_query_female=x_a_query_female,
            x_b_query_female=x_b_query_female,
        )


    def complain(self, manager, exact=False):
        male_predict = manager.model(self.x_query_male)
        female_predict = manager.model(self.x_query_female)
        Q_male = tf.reduce_mean(male_predict)
        Q_female = tf.reduce_mean(female_predict)
        margin = Q_male - Q_female
        AC = 0.0
        return ComplaintRet(AC=tf.stack([AC]), AQ=tf.stack([margin])) 

    def lc_complain(self, manager_a, manager_b):
        male_predict = manager_a.model(self.x_a_query_male) + manager_b.model(self.x_b_query_male)
        female_predict = manager_a.model(self.x_a_query_female) + manager_b.model(self.x_b_query_female)
        Q_male = tf.reduce_mean(male_predict)
        Q_female = tf.reduce_mean(female_predict)
        margin = Q_male - Q_female
        AC = 0.0
        return ComplaintRet(AC=tf.stack([AC]), AQ=tf.stack([margin]))
    
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
        AC = 0.0
        ids = [(self.x_a_query[:, 0] == 0).numpy(), (self.x_a_query[:, 0] == 1).numpy()]
        agg = "avg"
        # avg reweight
        for i, single_ids in enumerate(ids):
            ids[i] = (single_ids / np.sum(single_ids)).reshape((-1,1))
        groupby = 2
        groupby_agg = "diff"
        return ComplaintRet(AC=AC, Ids=ids, Agg=agg, Groupby=groupby, Groupbyagg=groupby_agg) 
    
