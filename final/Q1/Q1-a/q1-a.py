import matplotlib.pyplot as plt
import matplotlib.image as img
import random as rd
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

def cifar10_plot(data, meta, im_idx=0):
    im = data[KEY_DATA][im_idx, :]
    
    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))

    print("shape: ", img.shape)
    print("label: ", data[KEY_LABELS][im_idx])
    print("category:", meta[KEY_LABEL_NAMES][data[KEY_LABELS][im_idx]])         
    
    plt.imshow(img) 
    plt.show()

def choose_indices(data_dict):
    ind_list = []
    while(len(ind_list)<indices_list_len):
        ind = rd.randint(0,len(data_dict[KEY_DATA])-1)
        while(ind in ind_list):
            ind = rd.randint(0,len(data_dict[KEY_DATA])-1)
        ind_list.append(ind)
    return ind_list

meta = unpickle("../data/batches.meta")
data_dict = unpickle("../data/data_batch_1")
ind_list = choose_indices(data_dict)

print("exit one image to proceed to the next")
for i in range(len(ind_list)):
    cifar10_plot(data_dict,meta,ind_list[i])


data_dict = unpickle("../data/test_batch")
ind_list = choose_indices(data_dict)

print("Now showing test set images")
for i in range(len(ind_list)):
    cifar10_plot(data_dict,meta,ind_list[i])