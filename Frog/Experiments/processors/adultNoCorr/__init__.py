from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from ..processor import ComplaintRet, Processor


# 6000 vs 25000
class AdultNoCorrProcessor(Processor):
    def __init__(self):
        # a_lso in a dict so that it is saved in the database
        # b_y the inherited insert method
        super().__init__()
        self.oe = OrdinalEncoder()
        df = pd.read_csv(f"../processors/adultNoCorr/data/adult_proxy_data.csv")
        data = self.oe.fit_transform(df)
        np.random.shuffle(data)
        x = data[:, 1:]
        y = data[:, 0]
#         y[(y == 0)] = -1 # taylor

        random_state = int(tf.random.uniform([], maxval=2**10, seed=1))
        x_male = x[x[:, 0] == 1]
        y_male = y[x[:, 0] == 1]
        x_female =  x[x[:, 0] == 0]
        y_female = y[x[:, 0] == 0]
        
        query_rate = 0.25
        x_train_male, x_query_male, y_train_male, y_query_male = train_test_split(
            x_male, y_male, test_size=query_rate, random_state=random_state,
        )
        
        x_train_female, x_query_female, y_train_female, y_query_female = train_test_split(
            x_female, y_female, test_size=query_rate, random_state=random_state,
        )
#         avg_sal_query_male = np.mean(y_query_male)
#         count_query_female = int(len(x_female) * query_rate)
        
#         index_female_1 = np.where(y_female == 1)[0]
#         index_female_0 = np.where(y_female == 0)[0]
        
#         selected_index_query_female=[index_female_1[0]]
#         index_female_1 = index_female_1[1:]
#         for i in range(count_query_female-1):
#             avg = np.mean(y_female[selected_index_query_female]) 
#             if avg > avg_sal_query_male:
#                 selected_index_query_female.append(index_female_0[0])
#                 index_female_0 = index_female_0[1:]
#             else:
#                 selected_index_query_female.append(index_female_1[0])
#                 index_female_1 = index_female_1[1:]
        
#         x_query_female = x_female[selected_index_query_female]
#         y_query_female = y_female[selected_index_query_female]
        
#         index_train_female = np.concatenate([index_female_0, index_female_1])
#         x_train_female = x_female[index_train_female]
#         y_train_female = y_female[index_train_female]
        
        x_train = np.concatenate([x_train_female,x_train_male])
        y_train = np.concatenate([y_train_female, y_train_male])
        x_query = np.concatenate([x_query_female,x_query_male])
        y_query = np.concatenate([y_query_female, y_query_male])
            
        
        x_train, x_test, y_train, y_test = train_test_split(
            x_train, y_train, test_size=0.1, random_state=random_state,
        )
        
#         x_test, x_query, y_test, y_query = train_test_split(
#             x_test, y_test, test_size=0.5, random_state=random_state,
#         )

#         x_test_male = x_test[x_test[:, 0] == 1]
#         x_test_female = x_test[x_test[:, 0] == 0]
        
#         x_query_male = x_query[x_query[:, 0] == 1]
#         x_query_female = x_query[x_query[:, 0] == 0]

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

#     def complain(self, manager, exact=True):
# #         xquery_male = tf.concat([self.x_a_query_male, self.x_b_query_male], 1)
# #         xquery_female = tf.concat([self.x_a_query_female, self.x_b_query_female], 1)
        
#         male_income = manager.model(self.x_query_male)
#         female_income = manager.model(self.x_query_female)

# #         male_income = manager.predict_proba(self.x_query_male, self.x_b_query_male)
# #         female_income = manager.predict_proba(self.x_a_query_female, self.x_b_query_female)

#         if exact:
#             male_income = tf.math.round(male_income)
#             female_income = tf.math.round(female_income)

#         Q_male = tf.reduce_mean(male_income)
#         Q_female = tf.reduce_mean(female_income)
#         margin = Q_male - Q_female

#         AC = 0.0
#         return ComplaintRet(AC=tf.stack([AC]), AQ=tf.stack([margin]))

    def complain(self, manager, exact=False):
        male_predict = manager.model(self.x_query_male)
        female_predict = manager.model(self.x_query_female)
        Q_male = tf.reduce_mean(male_predict)
        Q_female = tf.reduce_mean(female_predict)
        margin = Q_male - Q_female
        AC = 0.0
        return ComplaintRet(AC=tf.stack([AC]), AQ=tf.stack([margin])) 
    
    def test_complain(self, manager, exact=False):
        male_predict = manager.model(self.x_a_query_male, self.x_b_query_male)
        female_predict = manager.model(self.x_a_query_female, self.x_b_query_female)
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

    