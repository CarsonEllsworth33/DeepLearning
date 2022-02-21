import pandas
# importing Matplotlib and Numpy Packages
import numpy as np
import matplotlib.pyplot as plt


def graph(x_range,y_vals):
    # plot our list in X,Y coordinates
    plt.Axes.set_ylabel="Accuracy"
    plt.Axes.set_xlabel="Max Leaf Nodes"
    plt.plot(x_range,y_vals)
    plt.show()

def NBayes():
    pass


if(__name__=="__main__"):
    NBayes()