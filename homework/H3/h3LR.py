import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import math

def preprocess_data():
    df = pd.read_csv("abalone.data")
    col_name = "Sex"
    df.loc[df[col_name] == "M",col_name] =1
    df.loc[df[col_name] == "F",col_name] =0
    df.loc[df[col_name] == "I",col_name] =2
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.5,random_state=0)
    return X_train, X_test, y_train, y_test

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_data()
    lr = LogisticRegression(lr=.1, num_iter=300000)
    lr.fit(X_train,y_train)
    preds = lr.predict(X_test)
    print(preds)