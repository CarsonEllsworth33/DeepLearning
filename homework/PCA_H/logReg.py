from audioop import bias
from operator import mod
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.decomposition import PCA

class logReg():
    def __init__(self,feature_size,class_size):
        self.class_size = class_size
        self.biases = np.random.rand(1,class_size)
        self.weights = np.random.rand(feature_size,class_size)

    def _predict(self,X):
        #print(X.shape, self.weights.shape)
        z_mat = np.matmul(X,self.weights) + self.biases
        #print(z_mat.shape)
        p_mat = softmax(z_mat,axis=1)
        
        #y_mat = np.argmax(p_mat,axis=1)
        return p_mat

    def fit(self, X, Y,epochs=1000, learning_rate=.2):
        Y = Y.astype(int)
        loss_list = np.array([]) #initiating an empty array
        err_list = np.array([])
        for _ in range(epochs):
            probabilities = self._predict(X) # Calculates probabilities for each possible outcome
            err_list = np.append(err_list,self._error_rate(np.argmax(probabilities,axis=1),Y))
            CELoss = self._cost(probabilities, Y) # Calculates cross entropy loss for actual target and predictions
            loss_list = np.append(loss_list, CELoss) # Adds the CELoss value for the epoch to loss_list
            probabilities[np.arange(X.shape[0]),Y] -= 1 # Substract 1 from the scores of the correct outcome
            delt_weight = probabilities.T.dot(X) # gradient of loss w.r.t. weights
            delt_biases = np.sum(probabilities, axis = 0).reshape(-1,1) # gradient of loss w.r.t. biases
            self.weights -= (learning_rate * delt_weight).T
            self.biases -= (learning_rate * delt_biases).T
        return loss_list, err_list

    def _cost(self,prediction,actuals):
        n_samples = prediction.shape[0]
        CELoss = 0
        for sample, i in zip(prediction, actuals):
            CELoss += -np.log(sample[i])
        CELoss /= n_samples
        return CELoss

    def _accuracy(self,predictions,actuals):
        correct_pred = 0
        for i in range(len(actuals)):
            if(predictions[i]==actuals[i]):
                correct_pred+=1
        accuracy = correct_pred/len(actuals)*100
        return accuracy

    def _error_rate(self,predictions,actuals):
        acc = self._accuracy(predictions,actuals)
        acc /= 100
        return 1-acc
                
    def test(self,X,Y):
        res = np.argmax(self._predict(X),axis=1)
        acc = self._accuracy(res,Y)
        return acc
                


if __name__ == '__main__':
    import tensorflow as tf
    epoch = 1000
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
    #print("X Train shape: ",x_train.shape,"X Test shape: ",x_test.shape,"Y Train shape: ",y_train.shape,"Y Test shape: ",y_test.shape)
    classes=[]
    if(len(classes)==0):
            # traverse for all elements
            for elem in y_train:
                # check if exists in unique_list or not
                if elem not in classes:
                    classes.append(elem)
            classes.sort()
    #PCA setup code
    #pca_train=PCA(n_components=50)
    #x_train = pca_train.fit_transform(x_train)
    #pca_test=PCA(n_components=50)
    #x_test = pca_test.fit_transform(x_test)
    #print(x_train.shape)

    model = logReg(feature_size = x_train.shape[1],class_size = len(classes))
    loss_list,err_list = model.fit(x_train, y_train,epochs=epoch,learning_rate=.005)
    print("model after {} epochs is {}% accuracte".format(epoch,model.test(x_test,y_test)))

    # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    ax.plot(np.arange(0,epoch), err_list, label='error rate')  # Plot some data on the axes.
    ax.set_xlabel('x label')  # Add an x-label to the axes.
    ax.set_ylabel('Error')  # Add a y-label to the axes.
    ax.set_title("Error vs Epoch")  # Add a title to the axes.
    ax.legend();  # Add a legend.

    plt.show()