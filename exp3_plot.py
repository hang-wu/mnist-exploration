__author__ = 'Hang Wu'
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")
sns.set_context("paper", rc={"lines.linewidth": 1})

res = pd.read_pickle('out/exp3.dat')
g = sns.pointplot(x="noise", y="test_err", hue="scheme", data=res, join=True)
g.set(ylim=(0,1.))
g.get_figure().savefig('out/exp3.pdf')