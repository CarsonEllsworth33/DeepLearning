from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
k=3
df = pd.read_csv("../data/daily-min-temperatures.csv")

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
_train, _test, X_train, X_test = train_test_split(X,y,test_size=.2,random_state=0)
#print(X_train[:10])
print(len(X_train))

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


X_train, y_train = split_sequence(X_train,k)
# summarize the data
for i in range(5):
    print(X_train[i], y_train[i])
