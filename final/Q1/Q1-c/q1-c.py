from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np

KEY_BATCH_LABELS = b'batch_label'
KEY_LABELS = b'labels'   
KEY_DATA = b'data'     
KEY_FILENAMES = b'filenames'
KEY_LABEL_NAMES = b"label_names"
indices_list_len = 5

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar10_reshape(data):
    im = data
    
    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))
    return img

def x_train_join():
    data = []
    y_label = []
    for i in range(1,6):
        d_batch_str = "../data/data_batch_{}".format(i)
        data_dict = unpickle(d_batch_str)
        for i in range(len(data_dict[KEY_DATA])):
            data.append(cifar10_reshape(data_dict[KEY_DATA][i]))
            y_label.append(data_dict[KEY_LABELS][i])
    return np.array(data), np.array(y_label)

def x_test_join():
    data = []
    y_label = []
    data_dict = unpickle("../data/test_batch")
    for i in range(len(data_dict[KEY_DATA])):
        data.append(cifar10_reshape(data_dict[KEY_DATA][i]))
        y_label.append(data_dict[KEY_LABELS][i])
    
    return np.array(data),np.array(y_label)

x_train,y_train = x_train_join()
x_test, y_test = x_test_join()
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

epoch_num=20

######################################################################################
#2 layer model max pooling
model = Sequential()
#layer 1
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

#layer 2
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=epoch_num)
   
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(1,epoch_num+1),hist.history["accuracy"]) 
ax.set_xlabel("epoch number")   
ax.set_ylabel("model accuracy")
ax.grid(True) 
ax.set_title("2 layer model max pooling") 
fig.savefig("L2MPool")
ax.clear() 

######################################################################################
#2 layer model average pooling
model = Sequential()
#layer 1
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(AveragePooling2D())
#layer 2
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=epoch_num)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(1,epoch_num+1),hist.history["accuracy"]) 
ax.set_xlabel("epoch number")   
ax.set_ylabel("model accuracy")
ax.grid(True) 
ax.set_title("2 layer model average pooling") 
fig.savefig("L2APool")
ax.clear() 
######################################################################################
#3 layer model max pooling
model = Sequential()
#layer 1
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
#layer 2
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
#layer 3
model.add(Conv2D(16, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=epoch_num)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(1,epoch_num+1),hist.history["accuracy"]) 
ax.set_xlabel("epoch number")   
ax.set_ylabel("model accuracy")
ax.grid(True) 
ax.set_title("3 layer model max pooling") 
fig.savefig("L3MPool")
ax.clear() 

######################################################################################
#3 layer model average pooling
model = Sequential()
#layer 1
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(AveragePooling2D())
#layer 2
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D())
#layer 3
model.add(Conv2D(16, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=epoch_num)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(1,epoch_num+1),hist.history["accuracy"]) 
ax.set_xlabel("epoch number")   
ax.set_ylabel("model accuracy")
ax.grid(True) 
ax.set_title("3 layer model average pooling") 
fig.savefig("L3APool")
ax.clear() 

######################################################################################
#4 layer model max pooling
model = Sequential()
#layer 1
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(32,32,3)))
model.add(MaxPooling2D())
#layer 2
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(MaxPooling2D())
#layer 3
model.add(Conv2D(16, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
#layer 4
model.add(Conv2D(8, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=epoch_num)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(1,epoch_num+1),hist.history["accuracy"]) 
ax.set_xlabel("epoch number")   
ax.set_ylabel("model accuracy")
ax.grid(True) 
ax.set_title("4 layer model max pooling") 
fig.savefig("L4MPool")
ax.clear() 
######################################################################################
#4 layer model average pooling
model = Sequential()
#layer 1
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(32,32,3)))
model.add(AveragePooling2D())
#layer 2
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(AveragePooling2D())
#layer 3
model.add(Conv2D(16, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D())
#layer 4
model.add(Conv2D(8, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=epoch_num)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(1,epoch_num+1),hist.history["accuracy"]) 
ax.set_xlabel("epoch number")   
ax.set_ylabel("model accuracy")
ax.grid(True) 
ax.set_title("4 layer model average pooling") 
fig.savefig("L4APool")
ax.clear() 