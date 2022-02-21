import numpy as np  
import matplotlib.pyplot as plt  
import math



def graph():  
    a = 6
    b = 3
    c = 1
    fig = plt.figure()
 
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    
    # defining all 3 axes
    x = np.arange(0,1,.01,dtype=float)
    y = np.arange(0, 1, .01,dtype=float)
    z = (math.factorial(a+b+c)/(math.factorial(a)+math.factorial(b)+math.factorial(c)))*(np.power(x,a))*(np.power(y,b)) * (np.power((1-x-y),c)) #put L(theta) here
    
    # plotting
    ax.plot3D(x, y, z, 'green')
    ax.set_title('L(theta)')
    plt.show()

graph()