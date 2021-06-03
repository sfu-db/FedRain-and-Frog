import sys
sys.path.append(".")
sys.path.append("../")

import gurobipy
import time
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.metrics import (classification_report, f1_score, precision_score, recall_score)
from tqdm import tnrange, trange
from json import dumps, loads

from mlsql.influence import InfluenceRanker
from mlsql.fixer import AutoFixer
from mlsql.manager import ModelManagerLM
from mlsql.lc_protocol import aggregate
from models.simple_cnn import SimpleCNN
from models.logreg import LogReg
from models.linear_comb import LinearComb
from processors.diabetes_corruption import DiabetesCorrProcessor

import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

def master_debug(proc, manager_a, eval_b):
    K = int(len(proc.x_a_train) / 10)
    corrsel = tf.cast(tf.ones(proc.x_a_train.shape[0]), dtype='bool')
    fixer_a = AutoFixer(manager_a, corrsel, K)
    
    rank_time = np.array([0.0, 0.0, 0.0])
    train_time = np.array([0.0, 0.0, 0.0])
    AQs = []
    weighted_f1 = []
    AQ = complain_value(manager_a, proc.fl_complain, proc.x_a_query)
    f1 = master_f1(manager_a, proc.x_a_test, proc.y_test, "test")
    AQs.append(float(AQ))
    weighted_f1.append(f1)
    step_size = 10
    rain_k = int(np.ceil(K / step_size))

    for k in trange(0, rain_k):
        nfix = min(step_size, K - step_size * k)
        assert nfix > 0

        cpu_time, enc_time, com_time, _ = master_rank_fix(fixer_a, nfix, manager_a, proc.fl_complain, proc.x_a_query, eval_b)
        rank_time += np.asarray([cpu_time, enc_time, com_time])
        
        cpu_time, enc_time, com_time, eval_b = master_fit(manager_a, max_iter=1000, tol=1e-6, lr=0.5, print_value=True)
        train_time += np.asarray([cpu_time, enc_time, com_time])

        AQ = complain_value(manager_a, proc.fl_complain, proc.x_a_query)
        f1 = master_f1(manager_a, proc.x_a_test, proc.y_test, "test")
        AQs.append(float(AQ))
        weighted_f1.append(f1)

    print("Rank time: Cpu {}, Enc {}, Com {}".format(rank_time[0], rank_time[1], rank_time[2]))
    print("Retrain time: Cpu {}, Enc {}, Com {}".format(train_time[0], train_time[1], train_time[2]))
    print("Complain: ", AQs)
    print("F1: ", weighted_f1)
    
#     df_rain = pd.DataFrame({
#         "Complain": np.array(AQs),
#         "F1": np.array(weighted_f1),
#         "K": [0] + list(range(step_size, K + step_size, step_size)),
#         "Method": np.repeat("LC_Rain", len(AQs)),
#     })


def master_rank_fix(fixer_a, nfix, manager_a, complain, x_a_query, eval_b):
    cpu_time, enc_time, com_time, r = master_rank(manager_a, complain, x_a_query, eval_b)
    fixer_a.fix(r, nfix)
    return cpu_time, enc_time, com_time, r

def complain_value(manager_a, complain, x_a_query):
    ret = complain()
    assert ret.AC is not None or ret.PC is not None
    assert ret.Ids is not None
    assert ret.Agg is not None
    value_a = tf.squeeze(manager_a.model(x_a_query))
    agg_value_a = aggregate(value_a.numpy().reshape((-1,1)), manager_a, ret.Ids, ret.Agg, ret.Groupby, ret.Groupbyagg, ret.AC)
    enc_a_ids = manager_a.encrypt(ret.Ids, manager_a.public_key)
    if ret.Groupby is None:
        enc_b_groupby = manager_a.encrypt(0.0, manager_a.public_key_b)
    else:
        enc_b_groupby = manager_a.encrypt(float(ret.Groupby), manager_a.public_key_b)
    manager_a.master_send((ret.Agg, enc_b_groupby, ret.Groupbyagg, enc_a_ids))
    enc_a_agg_value_b = manager_a.master_recv()
    agg_value_b = manager_a.decrypt(enc_a_agg_value_b)
    return float(agg_value_a + agg_value_b)

def master_qgrads(manager_a, complain, x_a_query):
    enc_time = 0
    cpu_time = 0
    com_time = 0
    start_1 = time.time()
    ret = complain()
    assert ret.AC is not None or ret.PC is not None
    assert ret.Ids is not None
    assert ret.Agg is not None
    value_a = tf.squeeze(manager_a.model(x_a_query))
    egrads_a = manager_a.model.egrads(x_a_query, None)
    agg_egrads_a = aggregate(egrads_a.numpy(), manager_a, ret.Ids, ret.Agg, ret.Groupby, ret.Groupbyagg, None)
    agg_value_a = aggregate(value_a.numpy().reshape((-1,1)), manager_a, ret.Ids, ret.Agg, ret.Groupby, ret.Groupbyagg, ret.AC)
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
    enc_a_ids = manager_a.encrypt(ret.Ids, manager_a.public_key)
    if ret.Groupby is None:
        enc_b_groupby = manager_a.encrypt(0.0, manager_a.public_key_b)
    else:
        enc_b_groupby = manager_a.encrypt(float(ret.Groupby), manager_a.public_key_b)
    enc_a_agg_value_a = manager_a.encrypt(agg_value_a, manager_a.public_key)
    enc_b_agg_egrads_a = manager_a.encrypt(agg_egrads_a, manager_a.public_key_b)
    end_1 = time.time()
    enc_time += end_1 - middle_1

    manager_a.master_send((ret.Agg, enc_b_groupby, ret.Groupbyagg, enc_a_ids, enc_a_agg_value_a))
    enc_a_s, r1_enc_a_agg_egrads_b, enc_b_r1 = manager_a.master_recv()
    start_2 = time.time()
    com_time += start_2 - end_1
    
    s = manager_a.decrypt(enc_a_s)
    r1_agg_egrads_b = manager_a.decrypt(r1_enc_a_agg_egrads_b)
    r1_enc_b_agg_egrads_b = manager_a.encrypt(r1_agg_egrads_b, manager_a.public_key_b)
    middle_2 = time.time()
    enc_time += middle_2 - start_2
    
    sgn = s / np.abs(s)
    r3 = (np.random.random() + 0.5) / 20 # [0.025, 0.075)
    enc_b_agg_egrads_b = r1_enc_b_agg_egrads_b - enc_b_r1
    r3_enc_b_Q = sgn * r3 * np.concatenate((enc_b_agg_egrads_a, enc_b_agg_egrads_b))
    end_2 = time.time()
    cpu_time += end_2 - middle_2
    
    manager_a.master_send(r3_enc_b_Q)
    com_time += time.time() - end_2
    return cpu_time, enc_time, com_time

def master_hessian(manager_a, value_b):
    enc_time = 0
    cpu_time = 0
    com_time = 0
    start_1 = time.time()
    egrads_a = manager_a.egrads(hess=True)
    H_aa = manager_a.hessian(True, other_party=value_b)
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
    r4_enc_b_egrads_b = manager_a.master_recv()
    start_2 = time.time()
    com_time += start_2 - middle_1
    
    enc_b_H_prime_ab = np.mean(np.expand_dims(egrads_a, axis=2) * np.expand_dims(r4_enc_b_egrads_b, axis=1), axis=0)
    middle_2 = time.time()
    cpu_time += middle_2 - start_2
    
    enc_b_sum_egrads_a = manager_a.encrypt(np.mean(egrads_a, axis=0), manager_a.public_key_b)
    enc_b_H_aa = manager_a.encrypt_from_tensor(H_aa, manager_a.public_key_b)
    end_2 = time.time()
    enc_time += end_2 - middle_2
    
    manager_a.master_send((enc_b_H_aa, enc_b_H_prime_ab, enc_b_sum_egrads_a))
    enc_a_ihvps_a = manager_a.master_recv()
    start_3 = time.time()
    com_time += start_3 - end_2
    
    ihvps_a = manager_a.decrypt_to_tensor(enc_a_ihvps_a)
    enc_time += time.time() - start_3
    return cpu_time, enc_time, com_time, ihvps_a

def master_rank(manager_a, complain, x_a_query, eval_b):
    cpu_time, enc_time, com_time = 0, 0, 0
    q_cpu_time, q_enc_time, q_com_time = master_qgrads(manager_a, complain, x_a_query)
    h_cpu_time, h_enc_time, h_com_time, ihvps_a = master_hessian(manager_a, eval_b)
    
    start_1 = time.time()
    I_a = -manager_a.egmul(ihvps_a)
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
    enc_b_I_a = manager_a.encrypt_from_tensor(I_a, manager_a.public_key_b)
    end_1 = time.time()
    enc_time += end_1 - middle_1
    
    manager_a.master_send(enc_b_I_a)
    enc_a_I_b = manager_a.master_recv()
    start_2 = time.time()
    com_time += start_2 - end_1
    
    I_b = manager_a.decrypt_to_tensor(enc_a_I_b)
    enc_time += time.time() - start_2
    
    cpu_time += (q_cpu_time + h_cpu_time)
    enc_time += (q_enc_time + h_enc_time)
    com_time += (q_com_time + h_com_time)
    return cpu_time, enc_time, com_time, tf.reshape(I_a + I_b, (-1,)) 

def master_fit(manager_a, max_iter=100, print_value=False, tol=1e-6, lr=0.5):
  enc_time = 0
  cpu_time = 0
  com_time = 0
  opt_a = tf.keras.optimizers.SGD(learning_rate=lr)
  last_loss = -np.inf
  for iteration in range(max_iter):
    start_1 = time.time()
    
    evaluate_a = manager_a.master_evaluate() # c1f1-y
    egrads_a = manager_a.egrads() # c1
    eval_a = evaluate_a.numpy()
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
#     enc_a_eval_a = manager_a.encrypt(eval_a, manager_a.public_key) # [c1f1-y]_a
    enc_b_eval_a = manager_a.encrypt(eval_a, manager_a.public_key_b) # [c1f1-y]_b
    end_1 = time.time()
    enc_time += end_1 - middle_1
    
#     manager_a.master_send(enc_a_eval_a)
    manager_a.master_send(enc_b_eval_a)
#     enc_b_eval_b = manager_a.master_recv()
    enc_a_eval_b = manager_a.master_recv()
    start_2 = time.time()
    com_time += start_2 - end_1
    
    eval_b = manager_a.decrypt(enc_a_eval_b)
    middle_2 = time.time()
    enc_time += middle_2 - start_2
    
#     enc_b_grads_a = (egrads_a.numpy() * (enc_b_eval_b + eval_a).reshape(-1,1)).mean(axis=0)
#     middle_2 = time.time()
#     cpu_time += middle_2 - start_2
    
#     manager_a.master_send(enc_b_grads_a)
#     enc_a_grads_b = manager_a.master_recv()
#     start_3 = time.time()
#     com_time += start_3 - middle_2
    
#     grads_b = manager_a.decrypt(enc_a_grads_b)
#     enc_b_grads_b = manager_a.encrypt(grads_b, manager_a.public_key_b)
#     middle_3 = time.time()
#     enc_time += middle_3 - start_3
    
#     manager_a.master_send(enc_b_grads_b)
#     enc_a_grads_a = manager_a.master_recv()
#     start_4 = time.time()
#     com_time += start_4 - middle_3
    
#     grads_a = manager_a.decrypt(enc_a_grads_a)
#     middle_4 = time.time()
#     enc_time += middle_4 - start_4
    
    grads_a = (egrads_a.numpy() * (eval_b + eval_a).reshape(-1,1)).mean(axis=0)
    current_loss = tf.reduce_mean((eval_a + eval_b) ** 2)
    cont = tf.math.abs(last_loss - current_loss) > tol
    last_loss = current_loss
    
#     cpu_time += time.time() - middle_4
    start_3 = time.time()
    cpu_time += start_3 - middle_2
    enc_b_cont = manager_a.encrypt(float(cont.numpy()), manager_a.public_key_b)
    middle_3 = time.time()
    enc_time += middle_3 - start_3
    manager_a.master_send(enc_b_cont)
    com_time += time.time() - middle_3
    print("Iter: {}, Loss: {}".format(iteration, last_loss))
    if not cont:
        break
    else:
        manager_a.update_gradient(opt_a, grads_a)
        
  if print_value:
    print("SGD loss:", last_loss)
    print("SGD steps:", iteration)
  return cpu_time, enc_time, com_time, eval_b

def master_f1(manager_a, inputs_a, labels, dataset_name):
    return f1_score(labels.numpy(), predict(manager_a, inputs_a, dataset_name), average='weighted')

def predict(manager_a, inputs_a, dataset_name):
    results_a = manager_a.model(inputs_a)
    manager_a.master_send(dataset_name)
    enc_a_results_b = manager_a.master_recv()
    results_b = manager_a.decrypt(enc_a_results_b)
    results = manager_a.model(inputs_a).numpy() + results_b
    return np.squeeze(results > 0.5).astype(np.int32)

def classify(manager_a, inputs_a, labels, dataset_name):
    return classification_report(labels.numpy(), predict(manager_a, inputs_a, dataset_name))

def report(manager_a, x_a_train, y_train, x_a_test, y_test):
    print("Model A name:", manager_a.model.model_name())
#     print("Model B name:", manager_b.model.model_name())
    print("On Training\n", classify(manager_a, x_a_train, y_train, 'train'))
    print("On Testing\n", classify(manager_a, x_a_test, y_test, 'test'))
    
if __name__ == "__main__":
    proc = DiabetesCorrProcessor()
    model_a = LinearComb(1)
    manager_a = ModelManagerLM(proc.x_a_train, proc.y_corr, model_a, 256)
    manager_a.master_init_socket()
    cpu_time, enc_time, com_time, eval_b = master_fit(manager_a, max_iter=2000, tol=1e-6, lr=0.5, print_value=True)
    print(cpu_time, enc_time, com_time)
    report(manager_a, proc.x_a_train, proc.y_corr, proc.x_a_test, proc.y_test)
    master_debug(proc, manager_a, eval_b)
    manager_a.socket.close()


