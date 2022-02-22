import pandas
# importing Matplotlib and Numpy Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def graph(x_range,y_vals):
    # plot our list in X,Y coordinates
    plt.Axes.set_ylabel="Accuracy"
    plt.Axes.set_xlabel="Max Leaf Nodes"
    plt.plot(x_range,y_vals)
    plt.show()

def preproc_data():
    #need to process data such that I have a joint dist table for each quality output
    df = pandas.read_csv("WineQT.csv")
    df.drop(["Id"],axis=1,inplace=True)
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)
    y_sums = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0,"10":0}

    for num in y_train:
        y_sums[str(num)]+=1
    
    for elem in y_sums:
        if(y_sums[elem] == 0):
            #make 1 hyp data
            X_train = np.append(X_train,[[1,1,1,1,1,1,1,1,1,1,1]],axis=0)
            y_train = np.append(y_train,[int(elem)])
            y_sums[elem]+=1
    
    return X_train,X_test,y_train,y_test,y_sums

def calc_prob_table(X_train, y_sums, y_train):
    total_rows = len(X_train)
    prob_y = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0,"10":0}
    
    for elem in prob_y:
        prob_y[elem] = y_sums[elem]/total_rows
    #print(prob_y)

    p_xstr = "p_x{}y{}"
    #this is going to be a 2dim dict that each xi|yk has its own sub dictionary 
    prob_xy = {p_xstr.format(i,j):{} for j in prob_y for i in range(1,X_train.shape[1]+1)}
    #print(X_train.shape[1])
    
    for j in range(len(y_train)):
        y = y_train[j]
        for i in range(1,X_train.shape[1]+1):
            #if the key ie Xi = value is not in prob table then add it
            if(X_train[j][i-1] not in prob_xy[p_xstr.format(i,y)]):#need to i-1 for 0 indexing
                prob_xy[p_xstr.format(i,y)]["{}".format(X_train[j][i-1])] = 1
            else:
                prob_xy[p_xstr.format(i,y)]["{}".format(X_train[j][i-1])] += 1

    for j in range(len(y_train)):
        y = y_train[j]
        for i in range(1,X_train.shape[1]+1):
            #if the key ie Xi = value is not in prob table then add it
            #print("dividing ",prob_xy[p_xstr.format(i,y)]["{}".format(X_train[j][i-1])], "by ", y_sums[str(y)])
            prob_xy[p_xstr.format(i,y)]["{}".format(X_train[j][i-1])] /= y_sums[str(y)]
    
    return prob_xy, prob_y # the full probability tables of P(Yk) and P(Xi|Yk)
    



    
def NBayes():
    X_train,X_test,y_train,y_test,y_sums = preproc_data()
    prob_xy, prob_y = calc_prob_table(X_train, y_sums, y_train)
    p_xstr = "p_x{}y{}"





if(__name__=="__main__"):
    NBayes()