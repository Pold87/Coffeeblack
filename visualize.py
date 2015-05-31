import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_palette("deep", desat=.6)
sns.set_context(rc= {"figure.figsize": (8,4)})

contexts = pd.read_csv("42.csv")

### Histograms

# plt.hist(contexts.Age, bins=25, color=sns.desaturate("indianred", 1))
#plt.show()


### Boxplot

sns.violinplot(contexts.Age, contexts.Referer)

#sns.distplot(contexts.Age)
#sns.set_style("whitegrid")
# sns.corrplot(contexts)

sns.plt.show()
