__author__ = 'Hang Wu'

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", rc={"lines.linewidth": 1})

import pandas as pd

#Plot Fig 1 in the report
res1 = pd.read_pickle('out/exp2_1.dat')

g = sns.pointplot(x="std", y="test_err", hue="scheme", data=res1,join=True, ci = None, markers=['o','v','x','+','h','D'])
g.set(ylim=(0,.8))

g.get_figure().savefig('out/exp2_1.pdf')



#Plot Fig 2 in the report
res2 = pd.read_pickle('out/exp2_2.dat')

res2['class'] = res2['class'].apply(int)
g = sns.factorplot(x="std",y="test_err",hue="class", col="scheme", data=res2.loc[res2['std'] < 600], ci = None, legend=False)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
g.set(ylim=(0,.7))

g.savefig('out/exp2_2.pdf')