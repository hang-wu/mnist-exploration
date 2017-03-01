__author__ = 'Hang Wu'

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", rc={"lines.linewidth": 1.5})

import numpy as np

F = np.load('out/exp1_losses.npz')
all_train_losses = F['all_train_losses']

g = sns.tsplot(data=all_train_losses, condition='Train', ci=[68, 95])
g.set(xlabel='mini-batches', ylabel='Negative Log Likelihood')
g.get_figure().savefig('out/exp1.pdf')
