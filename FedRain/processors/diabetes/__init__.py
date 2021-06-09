from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets

from mlsql import ComplaintRet, Processor

# 6000 vs 25000
class DiabetesProcessor(Processor):
    def __init__(self, mode: str, n: int, m: int, corr_rate: float = 0.3) -> None:
        assert mode in {"A", "B", "Full"}
        super().__init__()

        x, y = datasets.load_diabetes(return_X_y=True)

        rowids = np.arange(n) % x.shape[0]
        colids = np.arange(m) % x.shape[1]

        x = x[:, colids]
        x = x[rowids]
        y = y[rowids]

        x = preprocessing.normalize(x, norm="l2")
        y = np.asarray([(1 if i >= 140.5 else 0) for i in y])

        random_state = int(tf.random.uniform([], maxval=2 ** 10, seed=1))
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            x, y, test_size=0.2, random_state=random_state,
        )
        Xtest, Xquery, ytest, yquery = train_test_split(
            Xtest, ytest, test_size=0.5, random_state=random_state,
        )

        print(f"Dataset size: {n} x {m}")

        (candidates,) = np.where(np.array(ytrain) == 1)

        np.random.seed(1024)
        corrupt_idx = np.random.choice(
            candidates, size=int(len(candidates) * corr_rate), replace=False
        )
        corrsel = np.full((len(ytrain),), False)
        corrsel[corrupt_idx] = True
        ycorr = ytrain.copy()

        ycorr[corrsel] = 0

        if mode == "Full":
            self.set_tensor_variables(
                Xtrain=Xtrain,
                ytrain=ytrain,
                ycorr=ycorr,
                Xtest=Xtest,
                ytest=ytest,
                Xquery=Xquery,
                yquery=yquery,
                corrsel=corrsel,
            )
        else:
            a_features = x.shape[1] // 2

            if mode == "A":
                Xtrain_a = Xtrain[:, :a_features]
                Xtest_a = Xtest[:, 0:a_features]
                Xquery_a = Xquery[:, 0:a_features]

                self.set_tensor_variables(
                    Xtrain=Xtrain_a,
                    ytrain=ytrain,
                    ycorr=ycorr,
                    Xtest=Xtest_a,
                    ytest=ytest,
                    Xquery=Xquery_a,
                    yquery=yquery,
                    corrsel=corrsel,
                )

            else:
                Xtrain_b = Xtrain[:, a_features:]
                Xtest_b = Xtest[:, a_features:]
                Xquery_b = Xquery[:, a_features:]

                self.set_tensor_variables(
                    Xtrain=Xtrain_b, Xtest=Xtest_b, Xquery=Xquery_b,
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
        male_predict = manager_a.model(self.x_a_query_male) + manager_b.model(
            self.x_b_query_male
        )
        female_predict = manager_a.model(self.x_a_query_female) + manager_b.model(
            self.x_b_query_female
        )
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
        ids = [(self.x_a_query[:, 1] > 0).numpy(), (self.x_a_query[:, 1] < 0).numpy()]
        agg = "avg"
        # avg reweight
        for i, single_ids in enumerate(ids):
            ids[i] = (single_ids / np.sum(single_ids)).reshape((-1, 1))
        groupby = 2
        groupby_agg = "diff"
        return ComplaintRet(
            AC=AC, Ids=ids, Agg=agg, Groupby=groupby, Groupbyagg=groupby_agg
        )
