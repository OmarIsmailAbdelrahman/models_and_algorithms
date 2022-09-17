import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=sys.maxsize)

# set the max columns,row to none
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



df = pd.read_csv("train.csv")
print(df.info())
print(df.describe())
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(6, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.show()