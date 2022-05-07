from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

k=40
n_features = 1
units = 50
epoch_num = 300

df = pd.read_csv("../data/daily-min-temperatures.csv")

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
from sklearn.preprocessing import MinMaxScaler

#sc = MinMaxScaler(feature_range = (0, 1))
#y = sc.fit_transform(y.reshape(-1, 1))
#X_test = sc.fit_transform(X_test)
_train, _test, X_train, X_test = train_test_split(X,y,test_size=.2,random_state=0)

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
X_test, y_test = split_sequence(X_test,k)



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],n_features)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_train[:10])




regressor = Sequential()
regressor.add(LSTM(units, activation='relu',input_shape=(k,n_features)))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mse')

hist = regressor.fit(X_train, y_train, epochs = epoch_num)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(1,epoch_num+1),hist.history["loss"]) 
ax.set_xlabel("epoch number")   
ax.set_ylabel("model loss")
ax.grid(True) 
ax.set_title("LSTM loss vs epoch") 
fig.savefig("LSTM-loss")
ax.clear()


pred_temp = []
onePrint = False
for i in range(len(X_test)):
    xi = X_test[i]
    if not onePrint:
        print(xi)
    xi = np.array(xi).reshape((1,k,n_features))
    pred_temp.append(regressor.predict(xi,verbose=0)[0])
    if not onePrint:
        print("prediction: ", regressor.predict(xi,verbose=0))
        print(xi.shape,xi)
        onePrint = True
    
print("first pred temp: ",pred_temp[0])
pred_temp = np.array(pred_temp)
print(pred_temp.shape)
pred_temp = pred_temp.reshape(pred_temp.shape[0]*pred_temp.shape[1],1)

#pred_temp = pred_temp.reshape(pred_temp.shape[0],pred_temp.shape[1])
#pred_temp = sc.inverse_transform(pred_temp)




fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(pred_temp[:50],color='black',label='Prediced Temperature') 
ax.plot(y_test[:50],color='red',label='Actual Temperature') 
ax.set_xlabel("Time")   
ax.set_ylabel("Temperature")
ax.grid(True) 
ax.set_title("LTSM Temperature Prediction Model") 
fig.legend()
fig.savefig("LSTM-prediction")
ax.clear()