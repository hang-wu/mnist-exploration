# Dependencies
With Conda, simply run

```sh
conda install pytorch torchvision -c soumith
```

A more detailed installation guide of PyTorch for different environments can be found in [PyTorch]. Experiment codes are tested in Ubuntu 16.04 and plotting scripts are run in Mac OS X environment, with Python 3.5/ 3.6.

# Code Structures
 - net.py, utils.py - the builiding blocks of the experiments, mainly the network structures
 - exp_.py - the experiments codes for the three sections
 - exp_plot.py - the plotting scripts for each section
 - out/ - the intermediate folder storing for plots and tables in the report
 

# Running the Codes
Simpy invoke
```sh
python exp1.py
```
to reproduce the results.

   [PyTorch]: <http://pytorch.org/>