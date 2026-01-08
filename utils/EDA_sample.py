# Use it in Jupyter Notebook as the Begining of EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
sns.set_theme(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# If Skewness > 0.75, data doesn't normal distribution
sns.histplot(train[''], kde=True)
plt.title('Distribution of Student Exam Scores')
plt.show()
print(f"Skewness: {train[''].skew()}")

# Correlation Index
corrmat = train.corr(numeric_only=True)
top_corr_features = corrmat.index[abs(corrmat[""]) > 0.5]

plt.figure(figsize=(10, 10))
sns.heatmap(train[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.title("High Correlation Features")
plt.show()