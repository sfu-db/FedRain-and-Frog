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
from processors.diabetes import DiabetesProcessor
from mlsql.utils.ihvp import iHvp, iHvp_exp, iHvp_hess


import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

def slave_qgrads(manager_b, x_b_query):
    enc_time = 0
    cpu_time = 0
    com_time = 0
    start_1 = time.time()
    value_b = tf.squeeze(manager_b.model(x_b_query))
    egrads_b = manager_b.model.egrads(x_b_query, None)
    r1 = np.random.random(egrads_b.shape[1])
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
    enc_a_r1 = manager_b.encrypt(r1, manager_b.public_key_a)
    enc_b_r1 = manager_b.encrypt(r1, manager_b.public_key)
    end_1 = time.time()
    enc_time += end_1 - middle_1
    
    agg, enc_b_groupby, groupbyagg, enc_a_ids, enc_a_agg_value_a = manager_b.slave_recv()
    start_2 = time.time()
    com_time += start_2 - end_1
    
    groupby = manager_b.decrypt(enc_b_groupby)
    middle_2 = time.time()
    enc_time += middle_2 - start_2
    
    enc_a_agg_egrads_b = aggregate(egrads_b.numpy(), manager_b, enc_a_ids, agg, groupby, groupbyagg, None)
    enc_a_agg_value_b = aggregate(value_b.numpy().reshape((-1,1)), manager_b, enc_a_ids, agg, groupby, groupbyagg, None)
    r2 = (np.random.random() + 0.5) / 20 # [0.025, 0.075)
    enc_a_s = (enc_a_agg_value_a + enc_a_agg_value_b) * r2
    r1_enc_a_agg_egrads_b = enc_a_agg_egrads_b + enc_a_r1
    end_2 = time.time()
    cpu_time += end_2 - middle_2
    
    manager_b.slave_send((enc_a_s, r1_enc_a_agg_egrads_b, enc_b_r1))
    r3_enc_b_Q = manager_b.slave_recv()
    start_3 = time.time()
    com_time += start_3 - end_2
    
    r3_Q = manager_b.decrypt_to_tensor(r3_enc_b_Q)
    enc_time += time.time() - start_3
    
    return cpu_time, enc_time, com_time, r3_Q

def slave_hessian(manager_b, value_a, Q):
    enc_time = 0
    cpu_time = 0
    com_time = 0
    start_1 = time.time()
    egrads_b = manager_b.egrads(hess=True)
    H_bb = manager_b.hessian(False, other_party=value_a)
    r4 = np.random.random((1, egrads_b.shape[1]))
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
    r4_enc_b_egrads_b = manager_b.encrypt_from_tensor(egrads_b + r4)
    end_1 = time.time()
    enc_time += end_1 - middle_1
    
    manager_b.slave_send(r4_enc_b_egrads_b)
    enc_H_aa, enc_b_H_prime_ab, enc_b_sum_egrads_a = manager_b.slave_recv()
    start_2 = time.time()
    com_time += start_2 - end_1

    enc_b_H_ab = enc_b_H_prime_ab - r4 * enc_b_sum_egrads_a.reshape((-1,1))
    middle_2 = time.time()
    cpu_time += middle_2 - start_2
    
    H_ab = manager_b.decrypt_to_tensor(enc_b_H_ab)
    end_2 - time.time()
    enc_time += end_2 - middle_2
    
    H = tf.concat([tf.concat([H_aa, H_ab], 1), tf.concat([tf.transpose(H_ab), H_bb], 1)], 0)
    ihvps = iHvp_hess(H, Q)
    cpu_time += time.time() - end_2
    return cpu_time, enc_time, com_time, ihvps

def slave_rank(manager_b, x_b_query, eval_a):
    cpu_time, enc_time, com_time = 0, 0, 0
    q_cpu_time, q_enc_time, q_com_time, Q = slave_qgrads(manager_b, x_b_query)
    h_cpu_time, h_enc_time, h_com_time, ihvps = slave_hessian(manager_b, eval_a, Q)
    b_features = len(pack(manager_b.variables))
    
    start_1 = time.time()
    I_b = -manager_b.egmul(ihvps[-b_features:])
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
    enc_a_I_b = manager_b.encrypt(I_b, manager_b.public_key_a)
    end_1 = time.time()
    enc_time += end_1 - middle_1
    
    manager_b.slave_send(enc_a_I_b)
    enc_b_I_a = manager_b.slave_recv()
    start_2 = time.time()
    com_time += start_2 - end_1
    
    I_a = manager_b.decrypt(enc_b_I_a)
    enc_time += time.time() - start_2
    
    cpu_time += (q_cpu_time + h_cpu_time)
    enc_time += (q_enc_time + h_enc_time)
    com_time += (q_com_time + h_com_time)
    return cpu_time, enc_time, com_time, tf.reshape(I_a + I_b, (-1,)) 

def slave_fit(manager_b, max_iter=100, print_value=False, lr=0.5):
  enc_time = 0
  cpu_time = 0
  com_time = 0
  opt_b = tf.keras.optimizers.SGD(learning_rate=lr)
  last_loss = -np.inf
  for iteration in range(max_iter):
    start_1 = time.time()
    
    evaluate_b = manager_b.slave_evaluate() # c1f1-y
    egrads_b = manager_b.egrads() # c1
    eval_b = evaluate_b.numpy()
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
#     enc_b_eval_b = manager_b.encrypt(eval_b, manager_b.public_key) # [c2f2]_b
    enc_a_eval_b = manager_b.encrypt(eval_b, manager_b.public_key_a) # [c2f2]_a
    end_1 = time.time()
    enc_time += end_1 - middle_1
    
#     enc_a_eval_a = manager_b.slave_recv()
    enc_b_eval_a = manager_b.slave_recv()
#     manager_b.slave_send(enc_b_eval_b)
    manager_b.slave_send(enc_a_eval_b)
    start_2 = time.time()
    com_time += start_2 - end_1
    
    eval_a = manager_b.decrypt(enc_b_eval_a)
    middle_2 = time.time()
    enc_time += middle_2 - start_2
    
#     enc_a_grads_b = (egrads_b.numpy() * (enc_a_eval_a + eval_b).reshape(-1,1)).mean(axis=0)
#     middle_2 = time.time()
#     cpu_time += middle_2 - start_2
    
#     enc_b_grads_a = manager_b.slave_recv()
#     manager_b.slave_send(enc_a_grads_b)
#     start_3 = time.time()
#     com_time += start_3 - middle_2
    
#     grads_a = manager_b.decrypt(enc_b_grads_a)
#     enc_a_grads_a = manager_b.encrypt(grads_a, manager_b.public_key_a)
#     middle_3 = time.time()
#     enc_time += middle_3 - start_3
    
#     enc_b_grads_b = manager_b.slave_recv()
#     manager_b.slave_send(enc_a_grads_a)
#     start_4 = time.time()
#     com_time += start_4 - middle_3
    
#     grads_b = manager_b.decrypt(enc_b_grads_b)
#     middle_4 = time.time()
#     enc_time += middle_4 - start_4
    
    grads_b = (egrads_b.numpy() * (eval_a + eval_b).reshape(-1,1)).mean(axis=0)
    current_loss = tf.reduce_mean((eval_a + eval_b) ** 2)
#     cont = True #tf.math.abs(last_loss - current_loss) > tol
    last_loss = current_loss
    
#     cpu_time += time.time() - middle_4
    start_3 = time.time()
    cpu_time += start_3 - middle_2
    enc_b_cont = manager_b.slave_recv()
    middle_3 = time.time()
    com_time += middle_3 - start_3
    cont = manager_b.decrypt(enc_b_cont)
    enc_time += time.time() - middle_3
    print("Iter: {}, Loss: {}".format(iteration, last_loss))
    if not cont:
        break
    else:
        manager_b.update_gradient(opt_b, grads_b)
        
  if print_value:
    print("SGD loss:", last_loss)
    print("SGD steps:", iteration)
  return cpu_time, enc_time, com_time, eval_a

def predict(manager_b, proc):
    dataset_name = manager_b.slave_recv()
    if dataset_name == 'train':
        results_b = manager_b.model(proc.x_b_train)
    elif dataset_name == 'test':
        results_b = manager_b.model(proc.x_b_test)
    else:
        print("Error! No such dataset")
    enc_a_results_b = manager_b.encrypt_from_tensor(results_b, manager_b.public_key_a)
    manager_b.slave_send(enc_a_results_b)

if __name__ == "__main__":
    proc = DiabetesProcessor()
    model_b = LinearComb(1)
    manager_b = ModelManagerLM(proc.x_b_train, proc.y_train, model_b, 256)
    manager_b.slave_init_socket()
    cpu_time, enc_time, com_time, eval_a = slave_fit(manager_b, max_iter=10, lr=0.5, print_value=True)
    print(cpu_time, enc_time, com_time)
    cpu_time, enc_time, com_time, Q = slave_qgrads(manager_b, proc.x_b_query)
    print(Q)
    print(cpu_time, enc_time, com_time)
#     predict(manager_b, proc)
#     predict(manager_b, proc)
    