import numpy as np
import h5py
import pickle

def load_h5_data(input_file_path, attribute_list):
    file = h5py.File(input_file_path, 'r')
    result = []
    for attribute in attribute_list:
        result.append(file[attribute][:])
    file.close()
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)


def save_as_h5(img_id, token_caps, file_path):
    file = h5py.File(file_path, 'w')
    dt = h5py.special_dtype(vlen=np.dtype('int32'))
    file.create_dataset('img_id', data=img_id)
    file.create_dataset('captions', (len(img_id),), dtype=dt)
    file['captions'][...] = token_caps
    file.close()

def save_data_as_h5(img_id, img_feats, file_path):
    file = h5py.File(file_path, 'w')
    file.create_dataset('img_id', data=img_id)
    file.create_dataset('img_feats', data=img_feats)
    file.close()

def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(file_path):
    with open(file_path, 'rb') as handle:
        token = pickle.load(handle)
    return token