import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import math

#Creating a Function.
def normal_dist_func_gen(mean:float , sd:float):
    def pdf(x:float,_mn=mean,_std=sd)->float:
        try:
            prob_density = (np.pi*_std) * np.exp(-0.5*((x-_mn)/_std)**2)
            return prob_density
        except(ZeroDivisionError):
            #print("zero value encountered, sd: ",sd," mean: ",mean)
            prob_density = (np.pi*_std) * np.exp(-0.5*((x-_mn)/(_std+1000000))**2)
            return 0
    return pdf

# calc mean and std for X0
def sigma_x_m_2(m_y,X_train,y_train,xi,y):
    #xi=0
    i=0
    sig = 0
    for j in X_train:
        if(y_train[i] != y):
            i+=1
            continue
        else:
           sig += math.pow(j[xi]-m_y,2) 
        i+=1
    return sig
###################################

def preproc_data():
    #need to process data such that I have a joint dist table for each quality output
    df = pd.read_csv("abalone.data")

    col_name = "Sex"
    df.loc[df[col_name] == "M",col_name] =1
    df.loc[df[col_name] == "F",col_name] =0
    df.loc[df[col_name] == "I",col_name] =2
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.4,random_state=0)

    y_sums = {"{}".format(i):0 for i in range(0,30)}

    for num in y_train:
        y_sums[str(num)]+=1
    print(y_sums)
    return X_train,X_test,y_train,y_test,y_sums

def calc_prob_dist(X_train, y_sums, y_train):
    """
    This function calculates the probabilities needed to compute and classify using NB
    This is the training function
    @returns P(Xi|Yk), P(Yk) 
    """
    total_rows = len(X_train)
    prob_y = {"{}".format(i):0 for i in range(1,30)}
    
    for elem in prob_y:
        prob_y[elem] = y_sums[elem]/total_rows
    #print(prob_y)

    m_xstr = "m_x{}y{}"
    #this is going to be a 1dim dict that each xi|yk
    m_xy = {m_xstr.format(i,j):0 for j in prob_y for i in range(1,X_train.shape[1]+1)}

    std_xstr = "s_x{}y{}"
    std_xy = {std_xstr.format(i,j):0 for j in prob_y for i in range(1,X_train.shape[1]+1)}

    #this sums
    for j in range(len(y_train)):
        y = y_train[j]
        for i in range(1,X_train.shape[1]+1):
            m_xy[m_xstr.format(i,y)] += X_train[j][i-1]

    #this creates the mean
    for j in range(len(y_train)):
        y = y_train[j]
        for i in range(1,X_train.shape[1]+1):
            m_xy[m_xstr.format(i,y)] /= y_sums[str(y)]

    #print(m_xy)

    #this makes the std
    for j in range(len(y_train)):
        y = y_train[j]
        for i in range(1,X_train.shape[1]+1):
            n=y_sums[str(y)]
            std_xy[std_xstr.format(i,y)] = math.sqrt(
                sigma_x_m_2(m_xy[m_xstr.format(i,y)],X_train,y_train,i-1,y)/n
                )
    
    #print(std_xy)

    xiy_pdf = {"y{}".format(y):{} for y in range(1,30)}
    for yi in range(1,30):
        xiy_pdf["y{}".format(yi)] = {"x{}".format(xi+1):normal_dist_func_gen(mean=m_xy[m_xstr.format(xi+1,yi)],sd=std_xy[std_xstr.format(xi+1,yi)]) for xi in range(0,8)}

    return xiy_pdf

def GNB_Test(xiy_pdf,y_pdf,X_test,y_test):
    itter=0
    obs_act_list = []
    for data in X_test:
        act_result = y_test[itter]
        test_result = [1 for _ in range(1,30)]
        for yi in range(1,30):
            for xi in range(1,9):
                test_result[yi-1]*=xiy_pdf["y{}".format(act_result)]["x{}".format(xi)](data[xi-1]) #should pull correct dist prob
                
            test_result[yi-1]*=y_pdf(yi)
        
        obs_y = test_result.index(max(test_result)) + 1 #plus one since zero index
        obs_act_list.append((obs_y,act_result))
        itter+=1
    return obs_act_list
            
def process_results(res_list:list):
    num_match = 0
    for elem in res_list:
        if (elem[0] == elem[1]):
            num_match+=1
    return (num_match/len(res_list))*100

def GNBayes():
    X_train,X_test,y_train,y_test,y_sums = preproc_data()
    xiy_pdf = calc_prob_dist(X_train,y_sums,y_train)
    y_pdf = normal_dist_func_gen(mean=9.934,sd=3.224)
    results = GNB_Test(xiy_pdf,y_pdf,X_test,y_test)
    print("accuracy of {}".format(process_results(results)))
    

if __name__ == '__main__':
    GNBayes()