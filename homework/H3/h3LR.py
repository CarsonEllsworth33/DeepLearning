import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("abalone.data")

col_name = "Sex"
df.loc[df[col_name] == "M",col_name] =1
df.loc[df[col_name] == "F",col_name] =0
df.loc[df[col_name] == "I",col_name] =2
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.5,random_state=0)
#X = preprocessing.StandardScaler().fit(X_train)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print(clf.predict(X_test))