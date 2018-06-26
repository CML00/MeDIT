import h5py
import numpy as np

def WeightsInfo(data_path):
    '''
    Load the h5file and print all the weights.
    :param data_path: the path of the h5 file.
    :return:
    '''
    data_dict = h5py.File(data_path, 'r')
    for top_group_name in data_dict.keys():
        print(top_group_name)
        for group_name in data_dict[top_group_name].keys():
            print('    ' + group_name)
            for data_name in data_dict[top_group_name][group_name].keys():
                print('        ' + data_name + ': ' + str(np.shape(data_dict[top_group_name][group_name][data_name].value)))