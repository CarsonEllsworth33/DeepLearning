import pandas
# importing Matplotlib and Numpy Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

# Possible Bug: X_test has an Xi that is not in the X_train set due to using real nums, prob need to filter nums etc


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
    #need to simplify data so that there are no key errors during testing due to real val nums
    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            X_train[i][j] = math.ceil(X_train[i][j])
            

    for i in range(len(X_test)):
        for j in range(len(X_test[i])):
            X_test[i][j] = math.ceil(X_test[i][j])

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
    """
    This function calculates the probabilities needed to compute and classify using NB
    This is the training function
    @returns P(Xi|Yk), P(Yk) 
    """
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

def calc_new_y(prob_xy, prob_y, Xi_list, y):
    """
    returns the probability of a single P(Yk) PI P(Xi|Yk) for the argmax function
    @returns P(Yk) PI P(Xi|Yk)
    """
    p_xstr = "p_x{}y{}"
    pi_k = prob_y[str(y)]
    theta_ijk = 1
    i=1
    try:
        #print(Xi_list)
        for elem in Xi_list:
            #print(p_xstr.format(i,y),prob_xy[p_xstr.format(i,y)]," trying to access {}".format(elem))
            theta_ijk *= prob_xy[p_xstr.format(i,y)][str(elem)]
            i+=1

        return pi_k * theta_ijk
    except(KeyError):
        return .00001



def NB_test(prob_xy, prob_y, X_test, y_test):
    """
    Tests the accuracy of the NB model
    @return Observed Y, Actual Y
    """
    y_list = []
    for y in range(0,11): #max Yk is Y10
        y_list.append(calc_new_y(prob_xy, prob_y, X_test, y))
    
    print(y_list)
    #need to argmax here
    obs_y = max(y_list)
    act_y = y_test
    return obs_y, act_y

def NBayes():
    X_train,X_test,y_train,y_test,y_sums = preproc_data()
    prob_xy, prob_y = calc_prob_table(X_train, y_sums, y_train)
    for i in range(len(y_test)):
        res = NB_test(prob_xy,prob_y,X_test[i],y_test[i])
        p_error = np.abs(res[0]-res[1])/res[1]
        acc = 1 - p_error
        #print(acc)
    







if(__name__=="__main__"):
    NBayes()