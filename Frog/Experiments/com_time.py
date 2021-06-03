import json
import numpy as np

slave_send = np.array(json.load(open("slave_debug_send_log.json")))
slave_recv = np.array(json.load(open("slave_debug_recv_log.json")))
master_send = np.array(json.load(open("master_debug_send_log.json")))
master_recv = np.array(json.load(open("master_debug_recv_log.json")))

master_train_send = np.array(json.load(open("master_send_log.json")))
master_train_recv = np.array(json.load(open("master_recv_log.json")))
slave_train_send = np.array(json.load(open("slave_send_log.json")))
slave_train_recv = np.array(json.load(open("slave_recv_log.json")))

#print((master_train_recv - slave_train_send).sum())
#print((slave_train_recv - master_train_send).sum())
print((master_train_recv - slave_train_send).sum() + (slave_train_recv - master_train_send).sum())

print((master_recv - slave_send).sum() + (slave_recv - master_send).sum())
