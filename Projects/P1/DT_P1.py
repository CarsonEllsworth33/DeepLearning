import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# importing Matplotlib and Numpy Packages
import numpy as np
import matplotlib.pyplot as plt

MAX_LEAF_NODES = 100

def graph(x_range,y_vals):
    # plot our list in X,Y coordinates
    plt.Axes.set_ylabel="Accuracy"
    plt.Axes.set_xlabel="Max Leaf Nodes"
    plt.plot(x_range,y_vals)
    plt.show()

def DTree():
    global MAX_LEAF_NODES
    #data preprocessing
    df = pandas.read_csv("WineQT.csv")
    df.drop(["Id"],axis=1,inplace=True)
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    x_vals = [x for x in range(1,MAX_LEAF_NODES)]
    y_acc = []
    #create DTree with MAX_LEAF_NODES X and compare against varying sizes of DTrees
    for i in range(1,MAX_LEAF_NODES):
        dTree=DecisionTreeClassifier(random_state=0,max_leaf_nodes=i+1,splitter="random",criterion="gini")
        dTree.fit(X_train,y_train)
        print("DTree depth",dTree.get_depth())
        print("Number of Features used in Fit: ",dTree.n_features_in_)
        y_acc.append(0)
        for j in range(len(X_test)):
            y_obs = dTree.predict(X_test[j].reshape(1,-1))
            y_act = y_test[j]
            #print("Predicted Quality: {}\nActual Quality: {}".format(y_obs, y_act))
            acc = 1 - (np.abs(y_obs - y_act)/y_act)
            #print("Accuracy: {}\n\n".format(acc))
            y_acc[i-1]+=acc
        y_acc[i-1]/=len(X_test)
    
    graph(x_vals,y_acc)



    
    

if(__name__=="__main__"):
    DTree()