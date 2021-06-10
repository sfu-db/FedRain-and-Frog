# FedRain-and-Frog
Code of FedRain and Frog for VLDB 2022

## Reproduce FedRain Experiments

### Dependencies

Run `cd FedRain && poetry install` to install the python dependencies.

#### Reproduce Table 3

Goto `FedRain/scripts`, run `call_slave.py` in one docker container and `start_slave.py` in another 
container. Make sure to: 
1. change the slave address in the `call_slave.py` accordingly.
2. Set `enabled=True` for the LogFile class in both `call_slave.py`and `start_slave.py`.
3. Start a postgres server in docker and set the corresponding address when initializing the `LogFile` class in `call_slave.py`and `start_slave.py`.

Then run `call_slave.py` and the log will be populated to the database.
Run `FedRain/analysis.ipynb` to compute the time cost from the log.

#### Reproduce Figure 3

Similar to Table 3, but make sure to disable the log by setting `enabled=False` for the LogFile class in both `call_slave.py`and `start_slave.py`. This is due to logging will take a significant amount of time.
The end to end running time will be printed out once `call_slave.py` finished.


#### Reproduce Figure 4

Run `RecallK.ipynb` to reproduce the data for Figure 4.

## Reproduce Frog Experiments


### Dependencies

Run `cd Frog && pip install -r requirements.txt` to install the python dependencies.

#### Reproduce Table 3 & Figure 3

Goto `Experiments/processors/diabetes`, run `train_debug_master.py` in one docker container and `train_debug_slave.py` in another 
container. The computation and encryption time would be printed out.

Run `Experiments/com_time.py` to compute the time cost for communication.

For Figure 3, we duplicate the data points and features in 'Experiments/processors/diabetes/__init__.py'.


#### Reproduce Figure 4

Run `Experiments/adhoc/LinearComb_Diabetes.ipynb` and `Experiments/adhoc/LinearComb_BreastCancer.ipynb` to reproduce the data for Figure 4. The Recall@k plots are summarized in `Experiments/adhoc/LC_Corruption_Plot.ipynb`.

#### Reproduce Figure 5 & Table 5
Run `http://localhost:6008/notebooks/Frog/Experiments/adhoc/LinearComb_Adults_DisparateImpact.ipynb`.

