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
