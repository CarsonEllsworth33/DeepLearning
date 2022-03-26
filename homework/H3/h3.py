import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd

df = pd.read_csv("abalone.data")

col_name = "Sex"
df.loc[df[col_name] == "M",col_name] =1
df.loc[df[col_name] == "F",col_name] =0
df.loc[df[col_name] == "I",col_name] =2

print(df)
