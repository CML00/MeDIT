import os
import h5py
from random import shuffle
import numpy as np
from MeDIT.SaveAndLoad import LoadH5InfoForGenerate, LoadH5
from MeDIT.ArrayProcess import ExtractPatch
from MeDIT.DataAugmentor import AugmentParametersGenerator, DataAugmentor2D, random_2d_augment

def GetInputOutputNumber(case_folder):
    key_list = []
    for case in os.listdir(case_folder):
        case_path = os.path.join(case_folder, case)
        if case_path.endswith('.h5'):
            file = h5py.File(case_path, 'r')
            key_list = list(file.keys())
            file.close()
            break

    if len(key_list) == 0:
        print('there is no h5 file, check the folder path: ', case_folder)

    input_number, output_number = 0, 0
    for key in key_list:
        if 'input' in key:
            input_number += 1
        elif 'output' in key:
            output_number += 1
        else:
            print(key)

    if input_number > 0 and output_number > 0:
        return input_number, output_number
    else:
        print('Lack input or output: ', case_folder)
        return 0, 0

def AugmentDataList2D(data_list, augmentor):
    aug_data_list = []

    for data in data_list:
        aug_data = np.zeros_like(data)
        for channel_index in range(data.shape[-1]):
            aug_data[..., channel_index] = augmentor.Execute(data[..., channel_index])
        aug_data_list.append(aug_data)
    return aug_data_list

def CropDataList2D(data_list, input_shape):
    crop_data_list = []
    for data in data_list:
        crop_data = np.zeros((input_shape[0], input_shape[1], data.shape[-1]))
        for channel_index in range(data.shape[-1]):
            crop_data[..., channel_index], _ = ExtractPatch(data[..., channel_index], input_shape[:2])
        crop_data_list.append(crop_data)
    return crop_data_list

def AddOneSample(all_data_list, one_data_list):
    if len(all_data_list) != len(one_data_list):
        print('the number of all samples and the number of one data list is not same: ', len(all_data_list), len(one_data_list))
        return

    for index in range(len(one_data_list)):
        all_data_list[index].append(one_data_list[index])

def MakeKerasFormat(data_list, dtype=np.float32):
    if len(data_list) == 1:
        return np.asarray(data_list[0], dtype=dtype)
    else:
        format_data_list = []
        for one_input in data_list:
            format_data_list.append(np.asarray(one_input, dtype=dtype))
        return format_data_list

def ImageInImageOut2D(root_folder, input_shape, batch_size=8, augment_param={}):
    input_number, output_number = GetInputOutputNumber(root_folder)
    case_list = os.listdir(root_folder)

    input_list = [[] for index in range(input_number)]
    output_list = [[] for index in range(output_number)]

    param_generator = AugmentParametersGenerator()
    augmentor = DataAugmentor2D()

    while True:
        shuffle(case_list)
        for case in case_list:
            case_path = os.path.join(root_folder, case)
            if not case_path.endswith('.h5'):
                continue

            input_data_list, output_data_list = [], []
            file = h5py.File(case_path, 'r')
            for input_number_index in range(input_number):
                input_data_list.append(file['input_' + str(input_number_index)])
            for output_number_index in range(output_number):
                output_data_list.append(file['output_' + str(output_number_index)])

            param_generator.RandomParameters(augment_param)
            augmentor.SetParameter(param_generator.GetRandomParametersDict())

            input_data_list = AugmentDataList2D(input_data_list, augmentor)
            output_data_list = AugmentDataList2D(output_data_list, augmentor)

            input_data_list = CropDataList2D(input_data_list, input_shape)
            output_data_list = CropDataList2D(output_data_list, input_shape)

            AddOneSample(input_list, input_data_list)
            AddOneSample(output_list, output_data_list)

            if len(input_list[0]) >= batch_size:
                inputs = MakeKerasFormat(input_list)
                outputs = MakeKerasFormat(output_list)
                yield inputs, outputs
                input_list = [[] for index in range(input_number)]
                output_list = [[] for index in range(output_number)]

def ImageInImageOut2DTest(root_folder, input_shape):
    from MeDIT.Visualization import LoadWaitBar
    input_number, output_number = GetInputOutputNumber(root_folder)
    case_list = os.listdir(root_folder)

    input_list = [[] for index in range(input_number)]
    output_list = [[] for index in range(output_number)]

    for case in case_list:
        LoadWaitBar(len(case_list), case_list.index(case))
        case_path = os.path.join(root_folder, case)
        if not case_path.endswith('.h5'):
            continue

        input_data_list, output_data_list = [], []
        file = h5py.File(case_path, 'r')
        for input_number_index in range(input_number):
            input_data_list.append(file['input_' + str(input_number_index)])
        for output_number_index in range(output_number):
            output_data_list.append(file['output_' + str(output_number_index)])

        input_data_list = CropDataList2D(input_data_list, input_shape)
        output_data_list = CropDataList2D(output_data_list, input_shape)

        AddOneSample(input_list, input_data_list)
        AddOneSample(output_list, output_data_list)


    inputs = MakeKerasFormat(input_list)
    outputs = MakeKerasFormat(output_list)

    return inputs, outputs


def test():
    input_list, output_list = ImageInImageOut2D(r'z:\Data\CS_ProstateCancer_Detect_multicenter\JSPH_NIH_H5\3slice_input_2016-NIH\training',
                                                input_shape=[96, 96], batch_size=8, augment_param=random_2d_augment)

    input1, input2, input3 = input_list[0], input_list[1], input_list[2]
    input1 = np.concatenate((input1[..., 0], input1[..., 1], input1[..., 2]), axis=2)
    input2 = np.concatenate((input2[..., 0], input2[..., 1], input2[..., 2]), axis=2)
    input3 = np.concatenate((input3[..., 0], input3[..., 1], input3[..., 2]), axis=2)
    input1 = np.transpose(input1, (1, 2, 0))
    input2 = np.transpose(input2, (1, 2, 0))
    input3 = np.transpose(input3, (1, 2, 0))
    input_show = np.concatenate((input1, input2, input3), axis=0)
    roi = np.squeeze(output_list)
    roi = np.concatenate((np.zeros_like(roi), roi, np.zeros_like(roi)), axis=2)
    roi = np.concatenate((roi, roi, roi), axis=1)
    roi = np.transpose(roi, (1, 2, 0))
    from MeDIT.Visualization import Imshow3DArray
    Imshow3DArray(input_show, ROI=np.asarray(roi, dtype=np.uint8))

if __name__ == '__main__':
    test()