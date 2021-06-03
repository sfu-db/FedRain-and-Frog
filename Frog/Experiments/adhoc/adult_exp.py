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
from mlsql.models import SimpleCNN, LogReg
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
if not os.path.exists('adult_results/'):
    os.makedirs('adult_results/')

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
manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, 2))
manager.fit()
print("LogReg")
print("On Training Clean\n", classification_report(tf.argmax(proc.ytrain, axis=1).numpy(), manager.predict(proc.Xtrain).numpy()))
print("On Testing\n", classification_report(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy()))

# Init
manager0 = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
manager0.fit()
proc.post_init(manager0)

K = 2000
step_size = 10
corrsel = tf.cast(tf.ones(proc.ytrain.shape[0]), dtype='bool')

#@tf.function
#def change_model(manager, i):
#    egrad = manager.egrads(range_=[i])
#    ihvps = manager.iHvp(egrad)
#    n = tf.reduce_sum(manager.delta)
#    for var, ihvp in zip(manager.variables, ihvps):
#        var.assign_add(ihvp / n)

## No retraining IUP
#manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager.model.set_weights(manager0.model.get_weights())
#manager.delta = tf.Variable(manager0.delta.value(), name="delta")
#manager_check = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager_check.model.set_weights(manager0.model.get_weights())
#
#ranker = InfluenceRanker(manager=manager, on=proc.complain)
#fixer = AutoFixer(manager, corrsel, K)
#
## AQ is the margin of male - female
#AQs = []
#weighted_f1 = []
#AQs_re = []
#weighted_f1_re = []
#rank_time_iup = 0
#model_time_iup = 0
#for k in trange(0, K):
#    start = time.time()
#    scores = rankit(ranker)
#    ranks = tf.argsort(-scores)
##    tf.print(len(ranks))
##    tf.print(ranks[:20])
#    middle = time.time()
#    change_model(manager, ranks[0])
#    end = time.time()
#    
#    rank_time_iup += middle - start
#    model_time_iup += end - middle
#    
#    fixit(fixer, scores, 1)
#    manager_check.delta = tf.Variable(manager.delta.value(), name="delta")
#    manager_check.fit()
#    _, AQ, _, _ = proc.complain(manager_check)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager_check.predict(proc.Xtest).numpy(), average='weighted')
#    AQs_re.append(float(AQ))
#    weighted_f1_re.append(f1)
#    _, AQ, _, _ = proc.complain(manager)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#    AQs.append(float(AQ))
#    weighted_f1.append(f1)
#
#print("Rank_time:", rank_time_iup)
#print("Model_time:", model_time_iup)
#
#
#df_iup_re = pd.DataFrame({
#    "Complain": np.array(AQs_re),
#    "F1": np.array(weighted_f1_re),
#    "K": np.arange(len(AQs_re)) + 1,
#    "Method": np.repeat("IUp", len(AQs_re))
#})
#df_iup = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": np.arange(len(AQs)) + 1,
#    "Method": np.repeat("IUp", len(AQs))
#})
#time_iup = pd.DataFrame({
#    "Rank time": [rank_time_iup],
#    "Model time": [model_time_iup],
#    "Method": ["IUp"]
#})
#
#df_iup_re.to_csv('adult_results/df_iup_re.csv')
#df_iup.to_csv('adult_results/df_iup.csv')
#time_iup.to_csv('adult_results/time_iup.csv')

## Once
#manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager.model.set_weights(manager0.model.get_weights())
#manager.delta = tf.Variable(manager0.delta.value(), name="delta")
#ranker = InfluenceRanker(manager=manager, on=proc.complain)
#fixer = AutoFixer(manager, corrsel, K)
#
#AQs = []
#weighted_f1 = []
#rank_time_once = 0
#model_time_once = 0
#_, AQ, _, _ = proc.complain(manager)
#f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#AQs.append(float(AQ))
#weighted_f1.append(f1)
#
#start = time.time()
#rank_fix(ranker, fixer, K)
#middle = time.time()
#train(manager)
#end = time.time()
#
#rank_time_once += middle - start
#model_time_once += end - middle
#
#_, AQ, _, _ = proc.complain(manager)
#f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#AQs.append(float(AQ))
#weighted_f1.append(f1)
#
#print("Rank_time:", rank_time_once)
#print("Model_time:", model_time_once)
#
#df_once = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": [1, 2000],
#    "Method": np.repeat("Once", len(AQs)),
#})
#time_once = pd.DataFrame({
#    "Rank time": [rank_time_once],
#    "Model time": [model_time_once],
#    "Method": ["Once"]
#})
#
#df_once.to_csv('adult_results/df_once.csv')
#time_once.to_csv('adult_results/time_once.csv')

## Rain
#manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager.model.set_weights(manager0.model.get_weights())
#manager.delta = tf.Variable(manager0.delta.value(), name="delta")
#ranker = InfluenceRanker(manager=manager, on=proc.complain)
#fixer = AutoFixer(manager, corrsel, K)
#
#AQs = []
#weighted_f1 = []
#rank_time_rain = 0
#model_time_rain = 0
#_, AQ, _, _ = proc.complain(manager)
#f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#AQs.append(float(AQ))
#weighted_f1.append(f1)
#rain_k = int(np.ceil(K / step_size))
#for k in trange(0, rain_k):
#    nfix = min(step_size, K - step_size * k)
#    assert nfix > 0
#
#    start = time.time()
#    rank = rank_fix(ranker, fixer, nfix)
#    middle = time.time()
#    train(manager)
#    end = time.time()
#    
#    rank_time_rain += middle - start
#    model_time_rain += end - middle
#
#    _, AQ, _, _ = proc.complain(manager)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#    AQs.append(float(AQ))
#    weighted_f1.append(f1)
#
#print("Rank_time:", rank_time_rain)
#print("Model_time:", model_time_rain)
#
#df_rain = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": [1] + list(range(step_size, K + step_size, step_size)),
#    "Method": np.repeat("Rain", len(AQs)),
#})
#time_rain = pd.DataFrame({
#    "Rank time": [rank_time_rain],
#    "Model time": [model_time_rain],
#    "Method": ["Rain"]
#})
#
#df_rain.to_csv('adult_results/df_rain.csv')
#time_rain.to_csv('adult_results/time_rain.csv')

## Once Improved
#manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager.model.set_weights(manager0.model.get_weights())
#manager.delta = tf.Variable(manager0.delta.value(), name="delta")
#ranker = InfluenceRanker(manager=manager, on=proc.complain)
#fixer = AutoFixer(manager, corrsel, K)
#bpts = np.asarray([0, 400, 800, 1200, 1600, 2000])
#
#AQs = []
#weighted_f1 = []
#rank_time_oncei = 0
#model_time_oncei = 0
#_, AQ, _, _ = proc.complain(manager)
#f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#AQs.append(float(AQ))
#weighted_f1.append(f1)
#
#for nfix in bpts[1:] - bpts[:-1]:
#    start = time.time()
#    rank_fix(ranker, fixer, nfix)
#    middle = time.time()
#    train(manager)
#    end = time.time()
#    
#    rank_time_oncei += middle - start
#    model_time_oncei += end - middle
#
#    _, AQ, _, _ = proc.complain(manager)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#    AQs.append(float(AQ))
#    weighted_f1.append(f1)
#
#print("Rank_time:", rank_time_oncei)
#print("Model_time:", model_time_oncei)
#
#bpts[0] += 1
#df_oncei = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": bpts,
#    "Method": np.repeat("OnceI", len(AQs)),
#})
#time_oncei = pd.DataFrame({
#    "Rank time": [rank_time_oncei],
#    "Model time": [model_time_oncei],
#    "Method": ["OnceI"]
#})
#
#df_oncei.to_csv('adult_results/df_oncei.csv')
#time_oncei.to_csv('adult_results/time_oncei.csv')

## IUPI
#manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager.model.set_weights(manager0.model.get_weights())
#manager.delta = tf.Variable(manager0.delta.value(), name="delta")
#manager_check = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager_check.model.set_weights(manager0.model.get_weights())
#ranker = InfluenceRanker(manager=manager, on=proc.complain)
#fixer = AutoFixer(manager, corrsel, K)
#
#AQs = []
#weighted_f1 = []
#AQs_re = []
#weighted_f1_re = []
#rank_time_iupi = 0
#model_time_iupi = 0
#positions = []
#trainingKs = []
#K_list = []
#pbar = tqdm(total=K)
#n = proc.ytrain.shape[0]
#while pbar.n < K:
#    start = time.time()
#    scores = rankit(ranker)
#    ranks = tf.argsort(-scores)
#
#    indices = manager.to_original_index(ranks)
#    i2 = np.empty(n)
#    i2[:] = np.nan
#    i2[indices] = np.arange(len(indices)) + 1
#    positions.append(i2)
#
#    if len(positions) > 1:
#        topidx = np.argsort(positions[-1])[:10]
#        chs = (positions[-1] - positions[-2])[topidx]
#        need_train = (chs < -500).sum() > 3
#    else:
#        need_train = True
#
#    if need_train:
#        trainingKs.append(pbar.n)
#        fixit(fixer, scores, step_size)
#        middle = time.time()
#        train(manager)
#        end = time.time()
#        pbar.update(10)
#    else:
#        middle = time.time()
#        change_model(manager, ranks[0])
#        end = time.time()
#        fixer.fix(scores, 1)
#        pbar.update(1)
#    K_list.append(pbar.n)
#        
#    rank_time_iupi += middle - start
#    model_time_iupi += end - middle
#    
#    manager_check.delta = tf.Variable(manager.delta.value(), name="delta")
#    manager_check.fit()
#    _, AQ, _, _ = proc.complain(manager_check)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager_check.predict(proc.Xtest).numpy(), average='weighted')
#    AQs_re.append(float(AQ))
#    weighted_f1_re.append(f1)
#    _, AQ, _, _ = proc.complain(manager)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#    AQs.append(float(AQ))
#    weighted_f1.append(f1)
#
#print("Rank_time:", rank_time_iupi)
#print("Model_time:", model_time_iupi)
#
#df_iupi = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": K_list,
#    "Method": np.repeat("IUpI", len(AQs)),
#})
#df_iupi_re = pd.DataFrame({
#    "Complain": np.array(AQs_re),
#    "F1": np.array(weighted_f1_re),
#    "K": K_list,
#    "Method": np.repeat("IUpI_Re", len(AQs_re)),
#})
#time_iupi = pd.DataFrame({
#    "Rank time": [rank_time_iupi],
#    "Model time": [model_time_iupi],
#    "Method": ["IUpI"]
#})
#
#df_iupi.to_csv('adult_results/df_iupi.csv')
#df_iupi_re.to_csv('adult_results/df_iupi_re.csv')
#time_iupi.to_csv('adult_results/time_iupi.csv')

#@tf.function
#def change_model(manager, d): # d is a list
#    egrads = manager.egrads(range_=d)
#    egrad = [tf.reduce_sum(j, axis=0) for j in egrads]
#    ihvps = manager.iHvp(egrad)
#    n = tf.reduce_sum(manager.delta)
#    for var, ihvp in zip(manager.variables, ihvps):
#        var.assign_add(ihvp / n)

## No retraining IUP Step 10
#manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager.model.set_weights(manager0.model.get_weights())
#manager.delta = tf.Variable(manager0.delta.value(), name="delta")
#manager_check = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager_check.model.set_weights(manager0.model.get_weights())
#ranker = InfluenceRanker(manager=manager, on=proc.complain)
#fixer = AutoFixer(manager, corrsel, K)
#
## AQ is the margin of male - female
#AQs = []
#weighted_f1 = []
#AQs_re = []
#weighted_f1_re = []
#rank_time_iup_s10 = 0
#model_time_iup_s10 = 0
#for k in trange(0, int(K / step_size)):
#    start = time.time()
#    scores = tf.stop_gradient(rankit(ranker))
#    ranks = tf.argsort(-scores)
#    middle = time.time()
#
#    change_model(manager, ranks[:step_size])
#    end = time.time()
#    fixit(fixer, scores, step_size)  # update delta
#
#    rank_time_iup_s10 += middle - start
#    model_time_iup_s10 += end - middle
#    
#    manager_check.delta = tf.Variable(manager.delta.value(), name="delta")
#    manager_check.fit()
#    _, AQ, _, _ = proc.complain(manager_check)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager_check.predict(proc.Xtest).numpy(), average='weighted')
#    AQs_re.append(float(AQ))
#    weighted_f1_re.append(f1)
#    _, AQ, _, _ = proc.complain(manager)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#    AQs.append(float(AQ))
#    weighted_f1.append(f1)
#
#print("Rank_time:", rank_time_iup_s10)
#print("Model_time:", model_time_iup_s10)
#
#df_iup_s10_re = pd.DataFrame({
#    "Complain": np.array(AQs_re),
#    "F1": np.array(weighted_f1_re),
#    "K": (np.arange(len(AQs_re)) + 1) * 10,
#    "Method": np.repeat("IUp_Step10_Re", len(AQs_re)),
#})
#df_iup_s10 = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": (np.arange(len(AQs)) + 1) * 10,
#    "Method": np.repeat("IUp_Step10", len(AQs)),
#})
#time_iup_s10 = pd.DataFrame({
#    "Rank time": [rank_time_iup_s10],
#    "Model time": [model_time_iup_s10],
#    "Method": ["IUp_Step10"]
#})
#
#df_iup_s10.to_csv('adult_results/df_iup_s10.csv')
#df_iup_s10_re.to_csv('adult_results/df_iup_s10_re.csv')
#time_iup_s10.to_csv('adult_results/time_iup_s10.csv')


## IUPI Step 10
#manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager.model.set_weights(manager0.model.get_weights())
#manager.delta = tf.Variable(manager0.delta.value(), name="delta")
#manager_check = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager_check.model.set_weights(manager0.model.get_weights())
#ranker = InfluenceRanker(manager=manager, on=proc.complain)
#fixer = AutoFixer(manager, corrsel, K)
#
## AQ is the margin of male - female
#AQs = []
#weighted_f1 = []
#AQs_re = []
#weighted_f1_re = []
##influence_update = [None] * len(manager.variables)
#rank_time_iupi_s10 = 0
#model_time_iupi_s10 = 0
#positions = []
#trainingKs = []
#K_list = []
#pbar = tqdm(total=K)
#n = proc.ytrain.shape[0]
#while pbar.n < K:
#    start = time.time()
#    scores = tf.stop_gradient(rankit(ranker))
#    ranks = tf.argsort(-scores)
#
#    indices = manager.to_original_index(ranks)
#    i2 = np.empty(n)
#    i2[:] = np.nan
#    i2[indices] = np.arange(len(indices)) + 1
#    positions.append(i2)
#
#    if len(positions) > 1:
#        topidx = np.argsort(positions[-1])[:10]
#        chs = (positions[-1] - positions[-2])[topidx]
#        need_train = (chs < -500).sum() > 3
#    else:
#        need_train = True
#
#    if need_train:
#        trainingKs.append(pbar.n)
#        fixit(fixer, scores, step_size)
#        middle = time.time()
#        train(manager)
#        end = time.time()
#        pbar.update(10)
#    else:
#        middle = time.time()
#        change_model(manager, ranks[:step_size])
#        end = time.time()
#        fixit(fixer, scores, step_size)
#        pbar.update(10)
#    
#    rank_time_iupi_s10 += middle - start
#    model_time_iupi_s10 += end - middle
#    
#    manager_check.delta = tf.Variable(manager.delta.value(), name="delta")
#    manager_check.fit()
#    _, AQ, _, _ = proc.complain(manager_check)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager_check.predict(proc.Xtest).numpy(), average='weighted')
#    AQs_re.append(float(AQ))
#    weighted_f1_re.append(f1)
#    _, AQ, _, _ = proc.complain(manager)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#    AQs.append(float(AQ))
#    weighted_f1.append(f1)
#
#print("Rank_time:", rank_time_iupi_s10)
#print("Model_time:", model_time_iupi_s10)
#
#df_iupi_s10_re = pd.DataFrame({
#    "Complain": np.array(AQs_re),
#    "F1": np.array(weighted_f1_re),
#    "K": (np.arange(len(AQs_re)) + 1) * 10,
#    "Method": np.repeat("IUpI_Step10_Re", len(AQs_re)),
#})
#df_iupi_s10 = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": (np.arange(len(AQs)) + 1) * 10,
#    "Method": np.repeat("IUpI_Step10", len(AQs)),
#})
#time_iupi_s10 = pd.DataFrame({
#    "Rank time": [rank_time_iupi_s10],
#    "Model time": [model_time_iupi_s10],
#    "Method": ["IUpI_Step10"]
#})
#
#df_iupi_s10.to_csv('adult_results/df_iupi_s10.csv')
#df_iupi_s10_re.to_csv('adult_results/df_iupi_s10_re.csv')
#time_iupi_s10.to_csv('adult_results/time_iupi_s10.csv')


@tf.function
def change_model(manager, influence_update, d, hess_inv): # d is a list
    egrads = manager.egrads(range_=d)
    egrad = [tf.reduce_sum(j, axis=0) for j in egrads]
    ihvps = manager.iHvp(egrad, hess_inv)
    n = tf.reduce_sum(manager.delta)
    new_update = []
    for var, ihvp, ifu_update in zip(manager.variables, ihvps, influence_update):
        if ifu_update is not None:
            ifu_update += ihvp
        else:
            ifu_update = ihvp
        var.assign_add(ifu_update / n)
        new_update.append(tf.stop_gradient(ifu_update))
    return new_update

## No retraining IUP Step 10 Init
#manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager.model.set_weights(manager0.model.get_weights())
#manager.delta = tf.Variable(manager0.delta.value(), name="delta")
#manager_check = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
#manager_check.model.set_weights(manager0.model.get_weights())
#ranker = InfluenceRanker(manager=manager, on=proc.complain)
#fixer = AutoFixer(manager, corrsel, K)
#
## AQ is the margin of male - female
#AQs = []
#weighted_f1 = []
#AQs_re = []
#weighted_f1_re = []
#influence_update = [None] * len(manager.variables)
#rank_time_iup_s10_init = 0
#model_time_iup_s10_init = 0
#for k in trange(0, int(K / step_size)):
#    start = time.time()
#    scores = tf.stop_gradient(rankit(ranker))
#    ranks = tf.argsort(-scores)
#    middle = time.time()
#    
#
#    manager.model.set_weights(manager0.model.get_weights())  # clear
#    influence_update = change_model(manager, influence_update, ranks[:step_size])  # update
#    end = time.time()
#
#    fixit(fixer, scores, step_size) 
#    rank_time_iup_s10_init += middle - start
#    model_time_iup_s10_init += end - middle
#    
#    manager_check.delta = tf.Variable(manager.delta.value(), name="delta")
#    manager_check.fit()
#    _, AQ, _, _ = proc.complain(manager_check)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager_check.predict(proc.Xtest).numpy(), average='weighted')
#    AQs_re.append(float(AQ))
#    weighted_f1_re.append(f1)
#    _, AQ, _, _ = proc.complain(manager)
#    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
#    AQs.append(float(AQ))
#    weighted_f1.append(f1)
#
#
#print("Rank_time:", rank_time_iup_s10_init)
#print("Model_time:", model_time_iup_s10_init)
#
#df_iup_s10_init_re = pd.DataFrame({
#    "Complain": np.array(AQs_re),
#    "F1": np.array(weighted_f1_re),
#    "K": (np.arange(len(AQs_re)) + 1) * 10,
#    "Method": np.repeat("IUp_Step10_Init_Re", len(AQs_re)),
#})
#df_iup_s10_init = pd.DataFrame({
#    "Complain": np.array(AQs),
#    "F1": np.array(weighted_f1),
#    "K": (np.arange(len(AQs)) + 1) * 10,
#    "Method": np.repeat("IUp_Step10_Init", len(AQs)),
#})
#time_iup_s10_init = pd.DataFrame({
#    "Rank time": [rank_time_iup_s10_init],
#    "Model time": [model_time_iup_s10_init],
#    "Method": ["IUp_Step10_Init"]
#})
#
#df_iup_s10_init.to_csv('adult_results/df_iup_s10_init.csv')
#df_iup_s10_init_re.to_csv('adult_results/df_iup_s10_init_re.csv')
#time_iup_s10_init.to_csv('adult_results/time_iup_s10_init.csv')


# IUPI Step 10 Init
manager = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
manager.model.set_weights(manager0.model.get_weights())
manager.delta = tf.Variable(manager0.delta.value(), name="delta")
manager_check = ModelManager(proc.Xtrain, proc.ytrain, LogReg(proc, proc.ytrain.shape[1]))
manager_check.model.set_weights(manager0.model.get_weights())
ranker = InfluenceRanker(manager=manager, on=proc.complain)
fixer = AutoFixer(manager, corrsel, K)

#hess_list = []
#with tf.GradientTape(persistent=True) as tape:
#    cost = manager.eloss()
#    grads = tape.gradient(cost, manager.variables)
#    grads = pack(grads)
#    for grad in grads:
#        hess = tape.gradient(grad, manager.variables)
#        hess_list.append(pack(hess))
#hess = tf.stack(hess_list)
#hess_inv = tf.linalg.inv(hess) #tf.convert_to_tensor(np.linalg.pinv(hess))
#print("Hess:",hess)
#print("Hess_inv:",hess_inv)
#print("I:", tf.linalg.matmul(hess, hess_inv))
#manager.hess_inv = hess_inv
hess_inv = None

# AQ is the margin of male - female
AQs = []
weighted_f1 = []
AQs_re = []
weighted_f1_re = []
influence_update = [None] * len(manager.variables)
rank_time_iupi_s10_init = 0
model_time_iupi_s10_init = 0
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
        train(manager)
        end = time.time()
        pbar.update(10)
    else:
        middle = time.time()
        manager.model.set_weights(manager0.model.get_weights())  # clear
        influence_update = change_model(manager, influence_update, ranks[:step_size], hess_inv)  # update
#        change_model(manager, ranks[:step_size])
        end = time.time()
        fixit(fixer, scores, step_size)
        pbar.update(10)
    
    rank_time_iupi_s10_init += middle - start
    model_time_iupi_s10_init += end - middle
    
    manager_check.delta = tf.Variable(manager.delta.value(), name="delta")
    manager_check.fit()
    _, AQ, _, _ = proc.complain(manager_check)
    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager_check.predict(proc.Xtest).numpy(), average='weighted')
    AQs_re.append(float(AQ))
    weighted_f1_re.append(f1)
    _, AQ, _, _ = proc.complain(manager)
    f1 = f1_score(tf.argmax(proc.ytest, axis=1).numpy(), manager.predict(proc.Xtest).numpy(), average='weighted')
    AQs.append(float(AQ))
    weighted_f1.append(f1)

print("Rank_time:", rank_time_iupi_s10_init)
print("Model_time:", model_time_iupi_s10_init)

df_iupi_s10_init_re = pd.DataFrame({
    "Complain": np.array(AQs_re),
    "F1": np.array(weighted_f1_re),
    "K": (np.arange(len(AQs_re)) + 1) * 10,
    "Method": np.repeat("IUpI_Step10_Init_Re", len(AQs_re)),
})
df_iupi_s10_init = pd.DataFrame({
    "Complain": np.array(AQs),
    "F1": np.array(weighted_f1),
    "K": (np.arange(len(AQs)) + 1) * 10,
    "Method": np.repeat("IUpI_Step10_Init", len(AQs)),
})
time_iupi_s10_init = pd.DataFrame({
    "Rank time": [rank_time_iupi_s10_init],
    "Model time": [model_time_iupi_s10_init],
    "Method": ["IUpI_Step10_Init"]
})

df_iupi_s10_init.to_csv('adult_results/df_iupi_s10_init.csv')
df_iupi_s10_init_re.to_csv('adult_results/df_iupi_s10_init_re.csv')
time_iupi_s10_init.to_csv('adult_results/time_iupi_s10_init.csv')

