import numpy as np
import tensorflow as tf
import time
import timeit
from sklearn.metrics import classification_report, f1_score
from .utils.ihvp import iHvp, iHvp_exp, iHvp_hess, pack

def fit(manager_a, manager_b, max_iter=100, print_value=False, tol=1e-5, lr=0.1):
    enc_time = 0
    cpu_time = 0
    opt_a = tf.keras.optimizers.SGD(learning_rate=lr)
    opt_b = tf.keras.optimizers.SGD(learning_rate=lr)
    last_loss = -np.inf
    for iteration in range(max_iter):
        start = time.time()
        evaluate_a = manager_a.master_evaluate() # c1f1-y
        egrads_a = manager_a.egrads() # c1
        evaluate_b = manager_b.slave_evaluate() # c2f2
        egrads_b = manager_b.egrads() # c2

        middle = time.time()
        cpu_time += middle - start

        eval_a = evaluate_a.numpy()
    #     enc_a_eval_a = manager_a.encrypt(eval_a, manager_a.public_key) # [c1f1-y]_a
        eval_b = evaluate_b.numpy()
    #     enc_b_eval_b = manager_b.encrypt(eval_b, manager_b.public_key)# [c2f2]_b
    #     enc_a_grads_b = (egrads_b.numpy() * (enc_a_eval_a + eval_b).reshape(-1,1)).mean(axis=0)
    #     enc_b_grads_a = (egrads_a.numpy() * (enc_b_eval_b + eval_a).reshape(-1,1)).mean(axis=0)
    #     grads_a = manager_b.decrypt(enc_b_grads_a)
    #     grads_b = manager_a.decrypt(enc_a_grads_b)
        grads_a = (egrads_a.numpy() * (eval_b + eval_a).reshape(-1,1)).mean(axis=0)
        grads_b = (egrads_b.numpy() * (eval_a + eval_b).reshape(-1,1)).mean(axis=0)

        middle2 = time.time()
        enc_time += middle2 - middle

        manager_a.update_gradient(opt_a, grads_a)
        manager_b.update_gradient(opt_b, grads_b)
        current_loss = comb_loss(manager_a, manager_b)
        cont = tf.math.abs(last_loss - current_loss) > tol
        last_loss = current_loss

        cpu_time += time.time() - middle2
#       print(last_loss)
        if not cont:
            break
        
    if print_value:
        print("SGD loss:", last_loss)
        print("SGD steps:", iteration)
    return enc_time, cpu_time

def fit_test(manager_a, manager_b, max_iter=100, print_value=False, tol=1e-5, lr=0.1):
    enc_time = 0
    cpu_time = 0
    opt_a = tf.keras.optimizers.SGD(learning_rate=lr)
    opt_b = tf.keras.optimizers.SGD(learning_rate=lr)
    last_loss = -np.inf
    for iteration in range(max_iter):
        evaluate_a = manager_a.master_evaluate() # c1f1-y
        egrads_a = manager_a.egrads() # c1
        evaluate_b = manager_b.slave_evaluate() # c2f2
        egrads_b = manager_b.egrads() # c2

        eval_a = evaluate_a.numpy()
        eval_b = evaluate_b.numpy()
        grads_a = (egrads_a.numpy() * (eval_b + eval_a).reshape(-1,1)).mean(axis=0)
        grads_b = (egrads_b.numpy() * (eval_a + eval_b).reshape(-1,1)).mean(axis=0)
        manager_a.update_gradient(opt_a, grads_a)
        manager_b.update_gradient(opt_b, grads_b)
        current_loss = comb_loss(manager_a, manager_b)
        cont = tf.math.abs(last_loss - current_loss) > tol
        last_loss = current_loss

    #     print(last_loss)
        if not cont:
            break
        
    if print_value:
        print("SGD loss:", last_loss)
        print("SGD steps:", iteration)
    return enc_time, cpu_time

def comb_loss(manager_a, manager_b):
    return 0.5 * tf.reduce_mean((manager_a.master_evaluate() + manager_b.slave_evaluate()) ** 2)

def predict(manager_a, manager_b, inputs_a, inputs_b):
    results = manager_a.model(inputs_a) + manager_b.model(inputs_b)
    results = tf.squeeze(results > 0.5)
    return tf.cast(results, tf.int32)

def classify(manager_a, manager_b, inputs_a, inputs_b, labels):
    return classification_report(labels.numpy(), predict(manager_a, manager_b, inputs_a, inputs_b).numpy())

def f1(manager_a, manager_b, inputs_a, inputs_b, labels):
    return f1_score(labels.numpy(), predict(manager_a, manager_b, inputs_a, inputs_b).numpy(), average='weighted')

def report(manager_a, manager_b, x_a_train, x_b_train, y_train, x_a_test, x_b_test, y_test):
    print("Model A name:", manager_a.model.model_name())
    print("Model B name:", manager_b.model.model_name())
    print("On Training\n", classify(manager_a, manager_b, x_a_train, x_b_train, y_train))
    print("On Testing\n", classify(manager_a, manager_b, x_a_test, x_b_test, y_test))

def aggregate(value, manager, ids, agg, groupby, groupbyagg, AC):
    '''
    ids: index;
    agg: aggregation method (choices ["sum", "avg", "count?"])
    groupby: # of groupby categories, indicating IDs is "list" or "list of list"
    groupby_agg: groupby aggregation method (choice ["diff", "avg", "sum"])
    '''
    def single_agg(value, manager, ids, agg):
        ''' Seems redundant, but leave this function for future agg method updates'''
        if agg == "sum":
            return np.sum(value*ids, axis=0)
        elif agg == "avg":
            return np.sum(value*ids, axis=0)
        else:
            raise ValueError("Aggregation Not Implemented!")
            
    if groupby is not None:
        assert len(ids) == groupby
        single_agged = [single_agg(value, manager, single_ids, agg) for single_ids in ids]
        if groupbyagg == "diff":
            assert groupby == 2
            agged = single_agged[0] - single_agged[1]
        elif groupbyagg == "avg":
            agged = np.mean(single_agged)
        elif groupbyagg == "sum":
            agged = np.sum(single_agged)
        else:
            raise ValueError("Group Aggregation Not Implemented!")
    else:
        agged = single_agg(value, manager, ids, agg)
        
    if AC is not None:
        return agged - AC
    else:
        return agged
        
def complain_value(manager_a, manager_b, complain, query_data):
    ret = complain()
    assert ret.AC is not None or ret.PC is not None
    assert ret.Ids is not None
    assert ret.Agg is not None
    x_a_query, x_b_query = query_data
    value_a = tf.squeeze(manager_a.model(x_a_query))
    value_b = tf.squeeze(manager_b.model(x_b_query))
    agg_a = aggregate(value_a.numpy().reshape((-1,1)), manager_a, ret.Ids, ret.Agg, ret.Groupby, ret.Groupbyagg, ret.AC)
    agg_b = aggregate(value_b.numpy().reshape((-1,1)), manager_b, ret.Ids, ret.Agg, ret.Groupby, ret.Groupbyagg, None)
    return float(agg_a + agg_b)
    
def fl_qgrads(manager_a, manager_b, complain, query_data):
    ret = complain()
    assert ret.AC is not None or ret.PC is not None
    assert ret.Ids is not None
    assert ret.Agg is not None
    x_a_query, x_b_query = query_data
    value_a = tf.squeeze(manager_a.model(x_a_query))
    value_b = tf.squeeze(manager_b.model(x_b_query))
    egrads_a = manager_a.model.egrads(x_a_query, None)
    egrads_b = manager_b.model.egrads(x_b_query, None)
    
    agg_egrads_a = aggregate(egrads_a.numpy(), manager_a, ret.Ids, ret.Agg, ret.Groupby, ret.Groupbyagg, None)
    agg_value_a = aggregate(value_a.numpy().reshape((-1,1)), manager_a, ret.Ids, ret.Agg, ret.Groupby, ret.Groupbyagg, ret.AC)
    
    #enc
    enc_a_ids = manager_a.encrypt(ret.Ids, manager_a.public_key)
    enc_a_agg_value_a = manager_a.encrypt(agg_value_a, manager_a.public_key)
    enc_b_agg_egrads_a = manager_b.encrypt(agg_egrads_a, manager_b.public_key)
    
    enc_a_agg_egrads_b = aggregate(egrads_b.numpy(), manager_b, enc_a_ids, ret.Agg, ret.Groupby, ret.Groupbyagg, None)
    enc_a_agg_value_b = aggregate(value_b.numpy().reshape((-1,1)), manager_b, enc_a_ids, ret.Agg, ret.Groupby, ret.Groupbyagg, None)
    r1 = np.random.random(len(enc_a_agg_egrads_b))
    r2 = (np.random.random() + 0.5) / 20 # [0.025, 0.075)
    enc_a_s = (enc_a_agg_value_a + enc_a_agg_value_b) * r2
    
    #enc
    enc_a_r1 = manager_a.encrypt(r1, manager_a.public_key)
    enc_b_r1 = manager_b.encrypt(r1, manager_b.public_key)
    
    r1_enc_a_agg_egrads_b = enc_a_agg_egrads_b + enc_a_r1
    
    #dec/enc
    s = manager_a.decrypt(enc_a_s)
    r1_agg_egrads_b = manager_a.decrypt(r1_enc_a_agg_egrads_b)
    r1_enc_b_agg_egrads_b = manager_b.encrypt(r1_agg_egrads_b, manager_b.public_key)
    
    sgn = s / np.abs(s)
    r3 = (np.random.random() + 0.5) / 200 # [0.025, 0.075)
    enc_b_agg_egrads_b = r1_enc_b_agg_egrads_b - enc_b_r1
    r3_enc_b_Q = sgn * r3 * np.concatenate((enc_b_agg_egrads_a, enc_b_agg_egrads_b))
    
    #dec
    r3_Q = manager_b.decrypt_to_tensor(r3_enc_b_Q)
    return r3_Q

def hessian(manager_a, manager_b):
    egrads_a = manager_a.egrads(hess=True)
    egrads_b = manager_b.egrads(hess=True)
    value_a = manager_a.master_evaluate()
    value_b = manager_b.slave_evaluate()
    H_aa = manager_a.hessian(True, other_party=value_b)
    H_bb = manager_b.hessian(False, other_party=value_a)
    r4 = np.random.random((1, egrads_b.shape[1]))
    egrads_a = egrads_a.numpy()
    
    enc_b_egrads_b = manager_b.encrypt_from_tensor(egrads_b + r4, manager_b.public_key)
    
    enc_b_H_prime_ab = np.mean(np.expand_dims(egrads_a, axis=2) * np.expand_dims(enc_b_egrads_b, axis=1), axis=0)
    
    enc_b_sum_egrads_a = manager_b.encrypt(np.mean(egrads_a, axis=0), manager_b.public_key)

    enc_b_H_ab = enc_b_H_prime_ab - r4 * enc_b_sum_egrads_a.reshape((-1,1))
    
    H_ab = manager_b.decrypt_to_tensor(enc_b_H_ab)
    
    H = tf.concat([tf.concat([H_aa, H_ab], 1), tf.concat([tf.transpose(H_ab), H_bb], 1)], 0)
    return H

def rank(manager_a, manager_b, complain, query_data):
    Q = fl_qgrads(manager_a, manager_b, complain, query_data)
    H = hessian(manager_a, manager_b)
    ihvps = iHvp_hess(H, Q)
    a_features = len(pack(manager_a.variables))
    I_a = -manager_a.egmul(ihvps[:a_features])
    I_b = -manager_b.egmul(ihvps[a_features:])
    return tf.reshape(I_a + I_b, (-1,))    

### For debugging
def rank_test(manager_a, manager_b, complain, query_data):
    Q = qgrads_test(manager_a, manager_b, complain)
    ihvps = iHvp_exp(comb_loss, manager_a.variables + manager_b.variables, Q, manager_a=manager_a, manager_b=manager_b)
    I_a = -manager_a.egmul(pack([ihvps[1], ihvps[0], ihvps[2]]))
    I_b = -manager_b.egmul(pack([ihvps[4], ihvps[3], ihvps[5]]))
    return tf.reshape(I_a + I_b, (-1,))    

### For debugging
def qgrads_test(manager_a, manager_b, complain):
    with tf.GradientTape() as tape:
        AC, AQ, PC, PQ, _, _, _, _ = complain(manager_a, manager_b)
        query_loss = 0
        if AC is not None:
            query_loss += tf.norm(AC - AQ, 1)
        if PC is not None:
            query_loss += tf.nn.softmax_cross_entropy_with_logits(labels=PC,
                                                                  logits=PQ)
        qgrads = tape.gradient(query_loss, manager_a.variables + manager_b.variables)
    return qgrads

def rank_fix(fixer_a, fixer_b, nfix, manager_a, manager_b, complain, query_data):
    r = rank(manager_a, manager_b, complain, query_data)
#     r = rank_test(manager_a, manager_b, complain, query_data)    
    fixer_a.fix(r, nfix)
    fixer_b.fix(r, nfix)
    return r