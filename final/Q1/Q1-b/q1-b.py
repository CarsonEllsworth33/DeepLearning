from distutils.command.build import build
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.image as img
import random as rd
import numpy as np

KEY_BATCH_LABELS = b'batch_label'
KEY_LABELS = b'labels'   
KEY_DATA = b'data'     
KEY_FILENAMES = b'filenames'
KEY_LABEL_NAMES = b"label_names"
indices_list_len = 1000

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def choose_indices(data_dict):
    ind_list = []
    while(len(ind_list)<indices_list_len):
        ind = rd.randint(0,len(data_dict[KEY_DATA])-1)
        while(ind in ind_list):
            ind = rd.randint(0,len(data_dict[KEY_DATA])-1)
        ind_list.append(ind)
    return ind_list

def build_matrix(ind_list,data_dict) -> np.array:
    img_list = [data_dict[KEY_DATA][x] for x in ind_list]
    return np.array(img_list)


data_dict = unpickle("../data/data_batch_1")
ind_list = choose_indices(data_dict)
A_mat = build_matrix(ind_list,data_dict)
print(A_mat.shape)

pca = PCA(n_components=120)
pca.fit(A_mat)
print(pca.explained_variance_ratio_)
print("Sum of variance ratios: ", sum(pca.explained_variance_ratio_))
print("Data loss: ", 1-sum(pca.explained_variance_ratio_))