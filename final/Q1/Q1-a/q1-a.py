import matplotlib as plt

KEY_BATCH_LABELS = b'batch_label'
KEY_LABELS = b'labels'   
KEY_DATA = b'data'     
KEY_FILENAMES = b'filenames'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_dict = unpickle("../data/data_batch_1")


for key in data_dict:
    print(key)
print(data_dict[KEY_DATA].shape)