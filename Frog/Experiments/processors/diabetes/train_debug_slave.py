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
from mlsql.utils.utils import pack


import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

def slave_debug(proc, manager_b, eval_a):
    #K = int(len(proc.x_b_train) / 10)
    send_list = []
    recv_list = []
    K=5
    corrsel = tf.cast(tf.ones(proc.x_b_train.shape[0]), dtype='bool')
    fixer_b = AutoFixer(manager_b, corrsel, K)
    
    rank_time = np.array([0.0, 0.0, 0.0])
    train_time = np.array([0.0, 0.0, 0.0])
    AQs = []
    weighted_f1 = []
#    complain_value(manager_b, proc.x_b_query)
#    predict(manager_b, proc)
    step_size = 10
    rain_k = int(np.ceil(K / step_size))

    for k in trange(0, rain_k):
        nfix = min(step_size, K - step_size * k)
        assert nfix > 0

        cpu_time, enc_time, com_time, _, sd, rs = slave_rank_fix(fixer_b, nfix, manager_b, proc.x_b_query, eval_a)
        send_list += sd
        recv_list += rs
        rank_time += np.asarray([cpu_time, enc_time, com_time])
        
        cpu_time, enc_time, com_time, eval_a = slave_fit(manager_b, max_iter=1, lr=0.5, print_value=True)
        #train_time += np.asarray([cpu_time, enc_time, com_time])

#        complain_value(manager_b, proc.x_b_query)
#        predict(manager_b, proc)

    print("Rank time: Cpu {}, Enc {}, Com {}".format(rank_time[0], rank_time[1], rank_time[2]))
    print("Retrain time: Cpu {}, Enc {}, Com {}".format(train_time[0], train_time[1], train_time[2]))

    import json
    with open("slave_debug_send_log.json", 'w') as f:
        json.dump(send_list, f)
    with open("slave_debug_recv_log.json", 'w') as f:
        json.dump(recv_list, f)

#     df_rain = pd.DataFrame({
#         "Complain": np.array(AQs),
#         "F1": np.array(weighted_f1),
#         "K": [0] + list(range(step_size, K + step_size, step_size)),
#         "Method": np.repeat("LC_Rain", len(AQs)),
#     })

def slave_rank_fix(fixer_b, nfix, manager_b, x_b_query, eval_a):
    cpu_time, enc_time, com_time, r, sd, rs = slave_rank(manager_b, x_b_query, eval_a)
    fixer_b.fix(r, nfix)
    return cpu_time, enc_time, com_time, r, sd, rs

def complain_value(manager_b, x_b_query):
    value_b = tf.squeeze(manager_b.model(x_b_query))
    (agg, enc_b_groupby, groupbyagg, enc_a_ids), _ = manager_b.slave_recv()
    groupby = manager_b.decrypt(enc_b_groupby)
    enc_a_agg_value_b = aggregate(value_b.numpy().reshape((-1,1)), manager_b, enc_a_ids, agg, groupby, groupbyagg, None)
    manager_b.slave_send(enc_a_agg_value_b)

def slave_qgrads(manager_b, x_b_query):
    send_list, recv_list = [], []
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
    
    (agg, enc_b_groupby, groupbyagg, enc_a_ids, enc_a_agg_value_a), recv_time = manager_b.slave_recv()
    recv_list.append(recv_time)
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
    
    send_time = manager_b.slave_send((enc_a_s, r1_enc_a_agg_egrads_b, enc_b_r1))
    send_list.append(send_time)
    r3_enc_b_Q, recv_time = manager_b.slave_recv()
    recv_list.append(recv_time)
    start_3 = time.time()
    com_time += start_3 - end_2
    
    r3_Q = manager_b.decrypt_to_tensor(r3_enc_b_Q)
    enc_time += time.time() - start_3
    
    return cpu_time, enc_time, com_time, r3_Q, send_list, recv_list

def slave_hessian(manager_b, value_a, Q):
    send_list, recv_list = [], []
    enc_time = 0
    cpu_time = 0
    com_time = 0
    start_1 = time.time()
    egrads_b = manager_b.egrads(hess=True).numpy()
    H_bb = manager_b.hessian(False, other_party=value_a)
    b_features = len(pack(manager_b.variables))
    r4 = np.random.random((1, b_features))
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
 
    
    enc_b_r4 = manager_b.encrypt(r4, manager_b.public_key)
    end_1 = time.time()
    enc_time += end_1 - middle_1
    r4_enc_b_egrads_b = egrads_b + enc_b_r4
    final_1 = time.time()
    cpu_time += final_1 - end_1
    
    send_time = manager_b.slave_send(r4_enc_b_egrads_b)
    send_list.append(send_time)
    (enc_b_H_aa, enc_b_H_prime_ab, enc_b_sum_egrads_a), recv_time = manager_b.slave_recv()
    recv_list.append(recv_time)
    start_2 = time.time()
    com_time += start_2 - final_1

    enc_b_H_ab = enc_b_H_prime_ab - r4 * enc_b_sum_egrads_a.reshape((-1,1))
    middle_2 = time.time()
    cpu_time += middle_2 - start_2
    
    H_ab = manager_b.decrypt_to_tensor(enc_b_H_ab)
    #H_aa = manager_b.decrypt_to_tensor(enc_b_H_aa)
    H_aa = enc_b_H_aa
    end_2 = time.time()
    enc_time += end_2 - middle_2
    
    H = tf.concat([tf.concat([H_aa, H_ab], 1), tf.concat([tf.transpose(H_ab), H_bb], 1)], 0)
    ihvps = iHvp_hess(H, Q)
    start_3 = time.time()
    cpu_time += start_3 - end_2
    
    #enc_a_ihvps_a = manager_b.encrypt_from_tensor(ihvps[:-b_features], manager_b.public_key_a)
    enc_a_ihvps_a = ihvps[:-b_features]
    middle_3 = time.time()
    #enc_time += middle_3 - start_3
    
    send_time = manager_b.slave_send(enc_a_ihvps_a)
    send_list.append(send_time)
    com_time += time.time() - middle_3
    return cpu_time, enc_time, com_time, ihvps[-b_features:], send_list, recv_list

def slave_rank(manager_b, x_b_query, eval_a):
    send_list, recv_list = [], []
    cpu_time, enc_time, com_time = 0, 0, 0
    q_cpu_time, q_enc_time, q_com_time, Q, sd, rs = slave_qgrads(manager_b, x_b_query)
    send_list += sd
    recv_list += rs
    h_cpu_time, h_enc_time, h_com_time, ihvps_b, sd, rs = slave_hessian(manager_b, eval_a, Q)
    send_list += sd
    recv_list += rs

    start_1 = time.time()
    I_b = -manager_b.egmul(ihvps_b)
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
    #enc_a_I_b = manager_b.encrypt_from_tensor(I_b, manager_b.public_key_a)
    enc_a_I_b = I_b
    end_1 = time.time()
    #enc_time += end_1 - middle_1
    
    enc_b_I_a, recv_time = manager_b.slave_recv()
    recv_list.append(recv_time)
    send_time = manager_b.slave_send(enc_a_I_b)
    send_list.append(send_time)

    start_2 = time.time()
    com_time += start_2 - end_1
    
    I_a = enc_b_I_a
    #I_a = manager_b.decrypt_to_tensor(enc_b_I_a)
    #enc_time += time.time() - start_2
    
    cpu_time += (q_cpu_time + h_cpu_time)
    enc_time += (q_enc_time + h_enc_time)
    com_time += (q_com_time + h_com_time)
    return cpu_time, enc_time, com_time, tf.reshape(I_a + I_b, (-1,)), send_list, recv_list

def slave_fit(manager_b, max_iter=100, print_value=False, lr=0.5):
  enc_time = 0
  cpu_time = 0
  com_time = 0
  opt_b = tf.keras.optimizers.SGD(learning_rate=lr)
  last_loss = -np.inf
  send_list = []
  recv_list = []
  for iteration in range(max_iter):
    start_1 = time.time()
    
    evaluate_b = manager_b.slave_evaluate() # c1f1-y
    egrads_b = manager_b.egrads() # c1
    eval_b = evaluate_b.numpy()
    middle_1 = time.time()
    cpu_time += middle_1 - start_1
    
#     enc_b_eval_b = manager_b.encrypt(eval_b, manager_b.public_key) # [c2f2]_b
    #enc_a_eval_b = manager_b.encrypt(eval_b, manager_b.public_key_a) # [c2f2]_a
    enc_a_eval_b = eval_b
    end_1 = time.time()
    #enc_time += end_1 - middle_1
    
#     enc_a_eval_a = manager_b.slave_recv()
    enc_b_eval_a, recv_time = manager_b.slave_recv()
    recv_list.append(recv_time)
#     manager_b.slave_send(enc_b_eval_b)
    send_time = manager_b.slave_send(enc_a_eval_b)
    send_list.append(send_time)
    start_2 = time.time()
    com_time += start_2 - end_1
    
    #eval_a = manager_b.decrypt(enc_b_eval_a)
    eval_a = enc_b_eval_a
    middle_2 = time.time()
    #enc_time += middle_2 - start_2
    
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
    enc_b_cont, recv_time  = manager_b.slave_recv()
    recv_list.append(recv_time)
    middle_3 = time.time()
    com_time += middle_3 - start_3
    #cont = manager_b.decrypt(enc_b_cont)
    cont = enc_b_cont
    #enc_time += time.time() - middle_3
    end_3 = time.time()
    #print("Iter: {}, Loss: {}".format(iteration, last_loss))
    if not cont:
        break
    else:
        manager_b.update_gradient(opt_b, grads_b)
        cpu_time += time.time() - end_3

    import json
    with open("slave_send_log.json", 'w') as f:
        json.dump(send_list, f)
    with open("slave_recv_log.json", 'w') as f:
        json.dump(recv_list, f)
        
  if print_value:
    print("SGD loss:", last_loss)
    print("SGD steps:", iteration)
  return cpu_time, enc_time, com_time, eval_a

def predict(manager_b, proc):
    cpu_time = 0
    enc_time = 0
    com_time = 0
    send_list, recv_list = [], []
    dataset_name, recv_time = manager_b.slave_recv()
    recv_list.append(recv_time)
    start_1 = time.time()
    print(dataset_name)
    if dataset_name == 'train':
        results_b = manager_b.model(proc.x_b_train)
    elif dataset_name == 'test':
        results_b = manager_b.model(proc.x_b_test)
    else:
        print("Error! No such dataset")
    cpu_time += time.time() - start_1
    end_1 = time.time()
    #enc_a_results_b = manager_b.encrypt_from_tensor(results_b, manager_b.public_key_a)
    enc_a_results_b = results_b
    enc_time += time.time() - end_1
    end_2 = time.time()
    send_time = manager_b.slave_send(enc_a_results_b)
    send_list.append(send_time)
    com_time += time.time() - end_2
    print("inference time:", cpu_time, enc_time, com_time)

    import json
    with open("slave_infer_send_log.json", 'w') as f:
        json.dump(send_list, f)
    with open("slave_infer_recv_log.json", 'w') as f:
        json.dump(recv_list, f)

if __name__ == "__main__":
    proc = DiabetesProcessor()
    model_b = LinearComb(1)
    manager_b = ModelManagerLM(proc.x_b_train, proc.y_train, model_b, 256)
    manager_b.slave_init_socket()
    cpu_time, enc_time, com_time, eval_a = slave_fit(manager_b, max_iter=1000, lr=0.5, print_value=True)
#    print(cpu_time, enc_time, com_time)
#    predict(manager_b, proc)
    slave_debug(proc, manager_b, eval_a)
#     predict(manager_b, proc)
    
