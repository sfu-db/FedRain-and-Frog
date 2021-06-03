import sys
sys.path.append("../")
import gurobipy
from json import dumps, loads
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.metrics import (classification_report, f1_score, precision_score, recall_score)
from tqdm import tnrange, trange, tqdm
import tensorflow as tf

from mlsql import InfluenceRanker, SelfLossInfluenceRanker, AutoFixer, ModelManager, LossRanker, TiresiasRanker, multi_ambiguity_count
from mlsql.models.logreg import LogReg
from mlsql.models.simple_cnn import SimpleCNN, SimpleCNN1D
from mlsql.utils import setdiff1d
from mlsql.utils.pack import pack
from processors.adultNoCorr import AdultNoCorrProcessor

from itertools import groupby
from functools import partial

import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

import time
#import altair as alt
import os
RESULT_DIR = r'adult_results_deep/'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
        
@tf.function
def rank_fix(ranker, fixer, n):
    rank = ranker.predict()
    fixer.fix(rank, n)
    return rank

@tf.function
def rankit(ranker):
    rank = ranker.predict()
    return rank

@tf.function
def fixit(fixer, rank, n):
    fixer.fix(rank, n)

@tf.function
def train(manager):
    manager.fit()

seed = 2987429
proc = AdultNoCorrProcessor(seed)
print(proc.ytrain.shape)
manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
manager.fit()
print("LogReg")
print("On Training Clean\n", classification_report(tf.argmax(proc.ytrain, axis=1).numpy(), manager.predict(proc.Xtrain).numpy()))
print("On Testing\n", classification_report(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy()))

manager = ModelManager(proc.Xtrain, proc.ytrain, SimpleCNN1D(proc, proc.ytrain.shape[1], input_shape=[proc.Xtrain.shape[1], 1]))
manager.fit()
print("SimpleCNN1D")
print("On Training Clean\n", classification_report(tf.argmax(proc.ytrain, axis=1).numpy(), manager.predict(proc.Xtrain).numpy()))
print("On Testing\n", classification_report(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy()))

# Init
# manager0 = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
manager0 = ModelManager(proc.Xtrain, proc.ytrain, SimpleCNN1D(proc, proc.ytrain.shape[1], input_shape=[proc.Xtrain.shape[1], 1]))
manager0.fit()
proc.post_init(manager0)

K = 2000
step_size = 10
corrsel = tf.cast(tf.ones(proc.ytrain.shape[0]), dtype='bool')


@tf.function
def change_model(manager, d): # d is a list
   egrads = manager.egrads(range_=d)
   egrad = [tf.reduce_sum(j, axis=0) for j in egrads]
   ihvps = manager.iHvp(egrad)
   n = tf.reduce_sum(manager.delta)
   for var, ihvp in zip(manager.variables, ihvps):
       var.assign_add(ihvp / n)
   return egrad

# # Rain
# # manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
# manager = ModelManager(proc.Xtrain, proc.ytrain, SimpleCNN1D(proc, proc.ytrain.shape[1], input_shape=[proc.Xtrain.shape[1], 1]))
# manager.model.set_weights(manager0.model.get_weights())
# manager.delta = tf.Variable(manager0.delta.value(), name="delta")
# ranker = InfluenceRanker(manager=manager, on=proc.complain)
# fixer = AutoFixer(manager, corrsel, K)

# AQs = []
# weighted_f1 = []
# rank_time_rain = 0
# model_time_rain = 0
# _, AQ, _, _ = proc.complain(manager)
# f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
# AQs.append(float(AQ))
# weighted_f1.append(f1)
# rain_k = int(np.ceil(K / step_size))
# for k in trange(0, rain_k):
#    nfix = min(step_size, K - step_size * k)
#    assert nfix > 0

#    start = time.time()
#    rank = rank_fix(ranker, fixer, nfix)
#    middle = time.time()
# #    train(manager)
#    manager.fit(method="sgd")
#    end = time.time()
   
#    rank_time_rain += middle - start
#    model_time_rain += end - middle

#    _, AQ, _, _ = proc.complain(manager)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#    AQs.append(float(AQ))
#    weighted_f1.append(f1)

# print("Rank_time:", rank_time_rain)
# print("Model_time:", model_time_rain)

# df_rain = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": [1] + list(range(step_size, K + step_size, step_size)),
#    "Method": np.repeat("Rain", len(AQs)),
# })
# time_rain = pd.DataFrame({
#    "Rank time": [rank_time_rain],
#    "Model time": [model_time_rain],
#    "Method": ["Rain"]
# })

# df_rain.to_csv(RESULT_DIR + 'df_rain.csv')
# time_rain.to_csv(RESULT_DIR + 'time_rain.csv')

# # df_rain.to_csv(RESULT_DIR + 'df_rain_deep.csv')
# # time_rain.to_csv(RESULT_DIR + 'time_rain_deep.csv')


# # No retraining IUP Step 10
# manager = ModelManager(proc.Xtrain, proc.ytrain, SimpleCNN1D(proc, proc.ytrain.shape[1], input_shape=[proc.Xtrain.shape[1], 1]))
# manager.model.set_weights(manager0.model.get_weights())
# manager.delta = tf.Variable(manager0.delta.value(), name="delta")
# manager_check = ModelManager(proc.Xtrain, proc.ytrain, SimpleCNN1D(proc, proc.ytrain.shape[1], input_shape=[proc.Xtrain.shape[1], 1]))
# manager_check.model.set_weights(manager0.model.get_weights())
# ranker = InfluenceRanker(manager=manager, on=proc.complain)
# fixer = AutoFixer(manager, corrsel, K)

# # AQ is the margin of male - female
# AQs = []
# weighted_f1 = []
# AQs_re = []
# weighted_f1_re = []
# rank_time_iup_s10 = 0
# model_time_iup_s10 = 0
# for k in trange(0, int(K / step_size)):
#    start = time.time()
#    scores = tf.stop_gradient(rankit(ranker))
#    ranks = tf.argsort(-scores)
#    middle = time.time()

#    change_model(manager, ranks[:step_size])
#    end = time.time()
#    fixit(fixer, scores, step_size)  # update delta

#    rank_time_iup_s10 += middle - start
#    model_time_iup_s10 += end - middle
   
#    manager_check.delta = tf.Variable(manager.delta.value(), name="delta")
#    manager_check.fit(method="sgd")
#    _, AQ, _, _ = proc.complain(manager_check)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager_check.predict(proc.Xtest).numpy(), average='weighted')
#    AQs_re.append(float(AQ))
#    weighted_f1_re.append(f1)
#    _, AQ, _, _ = proc.complain(manager)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#    AQs.append(float(AQ))
#    weighted_f1.append(f1)

# print("Rank_time:", rank_time_iup_s10)
# print("Model_time:", model_time_iup_s10)

# df_iup_s10_re = pd.DataFrame({
#    "Complain": np.array(AQs_re),
#    "F1": np.array(weighted_f1_re),
#    "K": (np.arange(len(AQs_re)) + 1) * 10,
#    "Method": np.repeat("IUp_Step10_Re", len(AQs_re)),
# })
# df_iup_s10 = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": (np.arange(len(AQs)) + 1) * 10,
#    "Method": np.repeat("IUp_Step10", len(AQs)),
# })
# time_iup_s10 = pd.DataFrame({
#    "Rank time": [rank_time_iup_s10],
#    "Model time": [model_time_iup_s10],
#    "Method": ["IUp_Step10"]
# })

# df_iup_s10.to_csv(RESULT_DIR + 'df_iup_s10_deep.csv')
# df_iup_s10_re.to_csv(RESULT_DIR + 'df_iup_s10_deep_re.csv')
# time_iup_s10.to_csv(RESULT_DIR + 'time_iup_s10_deep.csv')


# IUPI Step 10
# manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
manager = ModelManager(proc.Xtrain, proc.ytrain, SimpleCNN1D(proc, proc.ytrain.shape[1], input_shape=[proc.Xtrain.shape[1], 1]))
manager.model.set_weights(manager0.model.get_weights())
manager.delta = tf.Variable(manager0.delta.value(), name="delta")
# manager_check = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
manager_check = ModelManager(proc.Xtrain, proc.ytrain, SimpleCNN1D(proc, proc.ytrain.shape[1], input_shape=[proc.Xtrain.shape[1], 1]))
manager_check.model.set_weights(manager0.model.get_weights())
ranker = InfluenceRanker(manager=manager, on=proc.complain)
fixer = AutoFixer(manager, corrsel, K)

# AQ is the margin of male - female
AQs = []
weighted_f1 = []
AQs_re = []
weighted_f1_re = []
egrad_list = []
loss_list = []
egrad = 0
rank_time_iupi_s10 = 0
model_time_iupi_s10 = 0
positions = []
trainingKs = []
K_list = []
pbar = tqdm(total=K)
n = proc.ytrain.shape[0]
while pbar.n < K:
   start = time.time()
   scores = tf.stop_gradient(rankit(ranker))
   ranks = tf.argsort(-scores)

   indices = manager.to_original_index(ranks)
   i2 = np.empty(n)
   i2[:] = np.nan
   i2[indices] = np.arange(len(indices)) + 1
   positions.append(i2)

   if len(positions) > 1:
       topidx = np.argsort(positions[-1])[:10]
       chs = (positions[-1] - positions[-2])[topidx]
       need_train = (chs < -500).sum() > 3
   else:
       need_train = True

   if need_train:
       trainingKs.append(pbar.n)
       fixit(fixer, scores, step_size)
       middle = time.time()
       manager.fit(method="sgd")
       end = time.time()
       pbar.update(10)
   else:
       middle = time.time()
       egrad = change_model(manager, ranks[:step_size])
       end = time.time()
       egrad = pack(egrad).numpy().mean()
       fixit(fixer, scores, step_size)
       pbar.update(10)
   
   rank_time_iupi_s10 += middle - start
   model_time_iupi_s10 += end - middle
   
   avg_loss = manager.loss().numpy()
   manager_check.delta = tf.Variable(manager.delta.value(), name="delta")
   manager_check.fit(method="sgd")
   _, AQ, _, _ = proc.complain(manager_check)
   f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager_check.predict(proc.Xtest).numpy(), average='weighted')
   AQs_re.append(float(AQ))
   weighted_f1_re.append(f1)
   _, AQ, _, _ = proc.complain(manager)
   f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
   AQs.append(float(AQ))
   weighted_f1.append(f1)
   egrad_list.append(egrad)
   loss_list.append(avg_loss)

print("Rank_time:", rank_time_iupi_s10)
print("Model_time:", model_time_iupi_s10)
print("Retraining:", trainingKs)

df_iupi_s10_re = pd.DataFrame({
   "Complain": np.array(AQs_re),
   "F1": np.array(weighted_f1_re),
   "K": (np.arange(len(AQs_re)) + 1) * 10,
   "Method": np.repeat("IUpI_Step10_Re", len(AQs_re)),
})
df_iupi_s10 = pd.DataFrame({
   "Complain": np.array(AQs),
   "F1": np.array(weighted_f1),
   "K": (np.arange(len(AQs)) + 1) * 10,
   "Egrad": np.array(egrad_list),
   "Loss": np.array(loss_list),
   "Method": np.repeat("IUpI_Step10", len(AQs)),
})
time_iupi_s10 = pd.DataFrame({
   "Rank time": [rank_time_iupi_s10],
   "Model time": [model_time_iupi_s10],
   "Method": ["IUpI_Step10"]
})

# df_iupi_s10.to_csv(RESULT_DIR + 'df_iupi_s10.csv')
# df_iupi_s10_re.to_csv(RESULT_DIR + 'df_iupi_s10_re.csv')
# time_iupi_s10.to_csv(RESULT_DIR + 'time_iupi_s10.csv')

df_iupi_s10.to_csv(RESULT_DIR + 'df_iupi_s10_deep.csv')
df_iupi_s10_re.to_csv(RESULT_DIR + 'df_iupi_s10_deep_re.csv')
time_iupi_s10.to_csv(RESULT_DIR + 'time_iupi_s10_deep.csv')