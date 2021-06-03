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

# 6000 vs 25000
class AdultProcessor(Processor):
    corruption_rate: float
    complaint_mode: str

    def __init__(self, seed: int, corruption_rate: float, complaint_mode: str):
        """
        Parameters
        ----------
        seed : int
        corruption_rate : float
        complaint_mode : str
            Candidates: ["gender", "age", "both"]
        """

        # Some easy checks
        assert 0 < corruption_rate <= 1
        # Complaint can either be "gender", "edu" or "both"
        assert complaint_mode in ["gender", "age", "both"]

        # Also in a dict so that it is saved in the database
        # By the inherited insert method
        super().__init__(
            seed=seed, corruption_rate=corruption_rate, complaint_mode=complaint_mode
        )

        cwd = Path(__file__).resolve().parent

        df = pd.read_csv(
            cwd / "data" / "adult.data",
            names=[
                "Age",
                "Workclass",
                "fnlwgt",
                "Education",
                "Education-Num",
                "Martial Status",
                "Occupation",
                "Relationship",
                "Race",
                "Gender",
                "Capital Gain",
                "Capital Loss",
                "Hours per week",
                "Country",
                "Income",
            ],
            na_values="?",
        )

        # Do preprocessing similar to the paper
        df["Age (decade)"] = df["Age"].apply(lambda x: np.floor(x / 10.0) * 10.0)

        # Limit education range
        df["Education Years"] = df["Education-Num"].apply(group_edu)

        # Limit age range
        df["Age (decade)"] = df["Age (decade)"].apply(age_cut)

        # Transform all that is non-white into 'minority'
        df["Race"] = df["Race"].apply(lambda x: x if x == " White" else "Minority")

        # Add binary income variable
        df["Income Binary"] = df["Income"].apply(lambda x: 1 if x == " >50K" else 0)

        # Keep only the columns we will use
        features = ["Age (decade)", "Education Years", "Gender"]
        output = ["Income Binary"]

        df_X = df[features]
        df_Y = df[output]

        X_features = pd.get_dummies(df_X).values.astype(np.float64)

        df_Xtrain, df_X_test, Xtrain, X_test, ytrain, y_test = train_test_split(
            df_X, X_features, df_Y, train_size=0.2, test_size=0.8, random_state=1
        )
        # Extract numpy arrays for labels
        ytrain = ytrain.values.ravel()
        y_test = y_test.values.ravel()

        # Find the corruptions
        noise_target = (
            (df_Xtrain["Gender"] == " Male")
            & (df_Xtrain["Age (decade)"] == 40)
            & (ytrain == 0)
        )
        candidates, = np.where(noise_target == 1)
        corruptions = int(candidates.shape[0] * self.corruption_rate)
        corrupt_idx = np.random.choice(candidates, size=corruptions, replace=False)

        # Implement the corruptions
        ycorr = ytrain.copy()
        ycorr[corrupt_idx] = 1
        corrsel = np.full((len(ytrain),), False)
        corrsel[corrupt_idx] = True
        self.corrsel = tf.constant(corrsel, name="corrsel")

        self.enc = OneHotEncoder(categories=[np.arange(2)], sparse=False)
        self.enc.fit(ytrain.reshape(-1, 1))
        ytrain = self.enc.transform(ytrain.reshape(-1, 1))
        ycorr = self.enc.transform(ycorr.reshape(-1, 1))
        y_test = self.enc.transform(y_test.reshape(-1, 1))

        first_partition = df_X_test["Gender"] == " Male"
        second_partition = df_X_test["Age (decade)"] == 40
        full_test = first_partition | second_partition

        self.set_tensor_variables(
            Xtrain=Xtrain,
            ytrain=ytrain,
            ycorr=ycorr,
            X_test_gender=X_test[first_partition, :],
            y_test_gender=y_test[first_partition, :],
            X_test_age=X_test[second_partition, :],
            y_test_age=y_test[second_partition, :],
            X_test_full=X_test[full_test, :],
            y_test_full=np.argmax(y_test[full_test, :], axis=1),  # Not one-hot encoded
        )

        full_test_indices = full_test[full_test].index

        first_indexes = []
        second_indexes = []

        for i, index in enumerate(full_test_indices):
            if first_partition[index]:
                first_indexes.append(i)
            if second_partition[index]:
                second_indexes.append(i)

        self.first_frozen = frozenset(first_indexes)
        self.second_frozen = frozenset(second_indexes)

    def complain(self, manager: ModelManager, exact: bool = False) -> ComplaintRet:
        AC, AQ, _, _ = complain_impl(
            self.X_test_gender,
            self.y_test_gender,
            self.X_test_age,
            self.y_test_age,
            manager,
            exact,
        )
        if self.complaint_mode == "gender":
            return ComplaintRet(AC=tf.reshape(AC[0], [1]), AQ=tf.reshape(AQ[0], [1]))
        elif self.complaint_mode == "age":
            return ComplaintRet(AC=tf.reshape(AC[1], [1]), AQ=tf.reshape(AQ[1], [1]))
        elif self.complaint_mode == "both":
            return ComplaintRet(AC=AC, AQ=AQ)
        else:
            raise Exception("Unknown mode")

def group_edu(x):
    if x <= 5:
        return "<6"
    elif x >= 13:
        return ">12"
    else:
        return x


def age_cut(x):
    if x >= 70:
        return ">=70"
    else:
        return x


@tf.function
def complain_impl(
    X_test_gender: tf.Tensor,
    y_test_gender: tf.Tensor,
    X_test_age: tf.Tensor,
    y_test_age: tf.Tensor,
    model: ModelManager,
    exact: bool = False,
) -> ComplaintRet:
    if exact:
        gender = tf.one_hot(model.predict(X_test_gender), 2)
        age = tf.one_hot(model.predict(X_test_age), 2)
    else:
        gender = model.predict_proba(X_test_gender)
        age = model.predict_proba(X_test_age)

    Q_gender = tf.reduce_mean(gender[:, 1])
    Q_age = tf.reduce_mean(age[:, 1])

    C_gender = tf.reduce_mean(y_test_gender[:, 1])
    C_age = tf.reduce_mean(y_test_age[:, 1])

    return ComplaintRet(AC=tf.stack([C_gender, C_age]), AQ=tf.stack([Q_gender, Q_age]))
