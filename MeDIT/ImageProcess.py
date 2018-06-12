import numpy as np
from skimage.transform import AffineTransform, warp
from scipy.ndimage.morphology import binary_dilation
import copy
import SimpleITK as sitk

from Normalize import IntensityTransfer


def FindBoundaryOfBinaryMask(image):
    kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    boundary = binary_dilation(input=image, structure=kernel, iterations=1) - image
    return boundary

### Transfer index to position #######################################################################################
def Index2XY(index, data_shape):
    y = np.mod(index, data_shape[1])
    x = np.floor_divide(index, data_shape[1])
    return [x, y]

def XY2Index(position, data_shape):
    return position[0] * data_shape[1] + position[1]

def Index2XYZ(index, data_shape):
    z = np.mod(index, data_shape[2])
    y = np.mod(np.floor_divide((index - z), data_shape[2]), data_shape[1])
    x = np.floor_divide(index, data_shape[2] * data_shape[1])
    return [x, y, z]

def XYZ2Index(position, data_shape):
    return position[0] * (data_shape[1] * data_shape[2]) + position[1] * data_shape[2] + position[2]

### Extract Patch from the image #######################################################################################
def ExtractPatches(image, patch_size, center_list):
    patch_list = []
    for center_point in center_list:
        patch = CatchPatch(image, patch_size, center_point)
        if patch != 0:
            patch_list.append(patch)
    return patch_list

def ExtractPatch(image, patch_size, center_point=[-1, -1], is_shift=True):
    patch_size = np.asarray(patch_size)
    if patch_size.shape == () or patch_size.shape == (1,):
        patch_size = np.array([patch_size[0], patch_size[0]])

    image_row, image_col = np.shape(image)
    catch_x_index = np.arange(patch_size[0] // 2, image_row - (patch_size[0] // 2))
    catch_y_index = np.arange(patch_size[1] // 2, image_col - (patch_size[1] // 2))

    if center_point == [-1, -1]:
        center_point[0] = image_row // 2
        center_point[1] = image_col // 2

    if patch_size[0] > image_row or patch_size[1] > image_col:
        print('The patch_size is larger than image shape')
        return np.array([])

    if center_point[0] < catch_x_index[0]:
        if is_shift:
            center_point[0] = catch_x_index[0]
        else:
            print('The center point is too close to the negative x-axis')
            return []
    if center_point[0] > catch_x_index[-1]:
        if is_shift:
            center_point[0] = catch_x_index[-1]
        else:
            print('The center point is too close to the positive x-axis')
            return []
    if center_point[1] < catch_y_index[0]:
        if is_shift:
            center_point[1] = catch_y_index[0]
        else:
            print('The center point is too close to the negative y-axis')
            return []
    if center_point[1] > catch_y_index[-1]:
        if is_shift:
            center_point[1] = catch_y_index[-1]
        else:
            print('The center point is too close to the positive y-axis')
            return []

    patch_row_index = [center_point[0] - patch_size[0] // 2, center_point[0] + patch_size[0] - patch_size[0] // 2]
    patch_col_index = [center_point[1] - patch_size[1] // 2, center_point[1] + patch_size[1] - patch_size[1] // 2]

    patch = image[patch_row_index[0]:patch_row_index[1], patch_col_index[0]:patch_col_index[1]]
    return patch

def ExtractBlock(image, patch_size, center_point=[-1, -1, -1], is_shift=False):
    if not isinstance(center_point, list):
        center_point = list(center_point)
    patch_size = np.asarray(patch_size)
    if patch_size.shape == () or patch_size.shape == (1,):
        patch_size = np.array([patch_size[0], patch_size[0], patch_size[0]])

    image_row, image_col, image_slice = np.shape(image)
    catch_x_index = np.arange(patch_size[0] // 2, image_row - (patch_size[0] // 2))
    catch_y_index = np.arange(patch_size[1] // 2, image_col - (patch_size[1] // 2))
    if patch_size[2] == image_slice:
        catch_z_index = [patch_size[2] // 2]
    else:
        catch_z_index = np.arange(patch_size[2] // 2, image_slice - (patch_size[2] // 2))

    if center_point == [-1, -1, -1]:
        center_point[0] = image_row // 2
        center_point[1] = image_col // 2
        center_point[2] = image_slice // 2

    if patch_size[0] > image_row or patch_size[1] > image_col or patch_size[2] > image_slice:
        print('The patch_size is larger than image shape')
        return np.array()

    if center_point[0] < catch_x_index[0]:
        if is_shift:
            center_point[0] = catch_x_index[0]
        else:
            print('The center point is too close to the negative x-axis')
            return np.array()
    if center_point[0] > catch_x_index[-1]:
        if is_shift:
            center_point[0] = catch_x_index[-1]
        else:
            print('The center point is too close to the positive x-axis')
            return np.array()
    if center_point[1] < catch_y_index[0]:
        if is_shift:
            center_point[1] = catch_y_index[0]
        else:
            print('The center point is too close to the negative y-axis')
            return np.array()
    if center_point[1] > catch_y_index[-1]:
        if is_shift:
            center_point[1] = catch_y_index[-1]
        else:
            print('The center point is too close to the positive y-axis')
            return np.array()
    if center_point[2] < catch_z_index[0]:
        if is_shift:
            center_point[2] = catch_z_index[0]
        else:
            print('The center point is too close to the negative z-axis')
            return np.array()
    if center_point[2] > catch_z_index[-1]:
        if is_shift:
            center_point[2] = catch_z_index[-1]
        else:
            print('The center point is too close to the positive z-axis')
            return np.array()
    #
    # if np.shape(np.where(catch_x_index == center_point[0]))[1] == 0 or \
    #     np.shape(np.where(catch_y_index == center_point[1]))[1] == 0 or \
    #     np.shape(np.where(catch_z_index == center_point[2]))[1] == 0:
    #     print('The center point is too close to the edge of the image')
    #     return []

    block_row_index = [center_point[0] - patch_size[0] // 2, center_point[0] + patch_size[0] - patch_size[0] // 2]
    block_col_index = [center_point[1] - patch_size[1] // 2, center_point[1] + patch_size[1] - patch_size[1] // 2]
    block_slice_index = [center_point[2] - patch_size[2] // 2, center_point[2] + patch_size[2] - patch_size[2] // 2]

    block = image[block_row_index[0]:block_row_index[1], block_col_index[0]:block_col_index[1], block_slice_index[0]:block_slice_index[1]]
    return block

def Crop2DImage(image, shape):
    if image.shape[0] >= shape[0]:
        center = image.shape[0] // 2
        if shape[0] % 2 == 0:
            new_image = image[center - shape[0] // 2: center + shape[0] // 2, :]
        else:
            new_image = image[center - shape[0] // 2: center + shape[0] // 2 + 1, :]
    else:
        new_image = np.zeros((shape[0], image.shape[1]))
        center = shape[0] // 2
        if image.shape[0] % 2 ==0:
            new_image[center - image.shape[0] // 2: center + image.shape[0] // 2, :] = image
        else:
            new_image[center - image.shape[0] // 2 - 1: center + image.shape[0] // 2, :] = image


    image = new_image
    if image.shape[1] >= shape[1]:
        center = image.shape[1] // 2
        if shape[1] % 2 == 0:
            new_image = image[:, center - shape[1] // 2: center + shape[1] // 2]
        else:
            new_image = image[:, center - shape[1] // 2: center + shape[1] // 2 + 1]
    else:
        new_image = np.zeros((image.shape[0], shape[1]))
        center = shape[1] // 2
        if image.shape[1] % 2 ==0:
            new_image[:, center - image.shape[1] // 2: center + image.shape[1] // 2] = image
        else:
            new_image[:, center - image.shape[1] // 2 - 1: center + image.shape[1] // 2] = image

    return new_image

def Crop3DImage(image, shape):
    if image.shape[0] >= shape[0]:
        center = image.shape[0] // 2
        if shape[0] % 2 == 0:
            new_image = image[center - shape[0] // 2: center + shape[0] // 2, :, :]
        else:
            new_image = image[center - shape[0] // 2: center + shape[0] // 2 + 1, :, :]
    else:
        new_image = np.zeros((shape[0], image.shape[1], image.shape[2]))
        center = shape[0] // 2
        if image.shape[0] % 2 ==0:
            new_image[center - image.shape[0] // 2: center + image.shape[0] // 2, :, :] = image
        else:
            new_image[center - image.shape[0] // 2 - 1: center + image.shape[0] // 2, :, :] = image

    image = new_image
    if image.shape[1] >= shape[1]:
        center = image.shape[1] // 2
        if image.shape[1] % 2 == 0:
            new_image = image[:, center - shape[1] // 2: center + shape[1] // 2, :]
        else:
            new_image = image[:, center - shape[1] // 2: center + shape[1] // 2 + 1, :]

    else:
        new_image = np.zeros((image.shape[0], shape[1], image.shape[2]))
        center = shape[1] // 2
        if shape[1] % 2 ==0:
            new_image[:, center - image.shape[1] // 2: center + image.shape[1] // 2, :] = image
        else:
            new_image[:, center - image.shape[1] // 2 - 1: center + image.shape[1] // 2, :] = image

    image = new_image
    if image.shape[2] >= shape[2]:
        center = image.shape[2] // 2
        if shape[2] % 2 == 0:
            new_image = image[:, :, center - shape[2] // 2: center + shape[2] // 2]
        else:
            new_image = image[:, :, center - shape[2] // 2: center + shape[2] // 2 + 1]
    else:
        new_image = np.zeros((image.shape[0], image.shape[1], shape[2]))
        center = shape[2] // 2
        if image.shape[2] % 2 ==0:
            new_image[:, :, center - image.shape[2] // 2: center + image.shape[2] // 2] = image
        else:
            new_image[:, :, center - image.shape[2] // 2 - 1: center + image.shape[2] // 2] = image

    return new_image

