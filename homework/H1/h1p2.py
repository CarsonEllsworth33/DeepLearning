import numpy as np  
import matplotlib.pyplot as plt  
import math



def graph1():  
    a = 500
    b = 0
    c = 500
    fig = plt.figure()
 
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    
    # defining all 3 axes
    x = np.arange(0, 1, .01,dtype=float)
    y = np.arange(0, 1, .01,dtype=float)
    z = (np.power(x,a))*(np.power(y,b)) * (np.power((1-x)*(1-y),c)) #put L(theta) here
    #z *= (math.factorial(a+b+c)/(math.factorial(a)+math.factorial(b)+math.factorial(c)))
    # plotting
    ax.plot3D(x, y, z, 'green')
    ax.set_title('L(theta)')
    plt.show()


def graph2():  
    a = 500 #alpha1
    b = 0 #alpha2      observed data
    c = 500 #alpha3

    d = 3 #beta1
    e = 3 #beta2       hypothetical data
    f = 3 #beta3
    fig = plt.figure()
 
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    
    # defining all 3 axes
    x = np.arange(0, 1, .01,dtype=float)
    y = np.arange(0, 1, .01,dtype=float)
    z = (np.power(x,a+d-1))*(np.power(y,b+e-1)) * (np.power((1-x)*(1-y),c+f-1)) #put L(theta) here
    #z *= (math.factorial(a+b+c)/(math.factorial(a)+math.factorial(b)+math.factorial(c)))
    # plotting
    ax.plot3D(x, y, z, 'green')
    ax.set_title('L(theta)')
    plt.show()

graph2()