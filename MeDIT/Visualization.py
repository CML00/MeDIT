from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import imread
import os

from MeDIT.Normalize import Normalize01
from MeDIT.ArrayProcess import Index2XY

def DrawBoundaryOfBinaryMask(image, ROI):
    '''
    show the image with ROIs
    :param image: the 2D image
    :param ROI: the binary ROI with same size of the image
    :return:
    '''
    plt.imshow(image, cmap='Greys_r')
    plt.contour(ROI, colors='y')
    plt.show()

def LoadWaitBar(total, progress):
    '''
    To show the waitbar for visulization
    :param total: the number of the total step
    :param progress: the number of the current processing
    :return:
    '''
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()

class IndexTracker(object):
    def __init__(self, ax, X, vmin, vmax, ROI):
        self.ax = ax

        self.X = X
        self.ROI = ROI
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin=vmin, vmax=vmax)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.slices - 1)
        self.update()

    def FindBoundaryOfROI(self):
        kernel = np.ones((3, 3))
        if isinstance(self.ROI, list):
            boundary_list = []
            for roi in self.ROI:
                boundary_list.append(binary_dilation(input=roi[:, :, self.ind], structure=kernel, iterations=1) - roi[:, :, self.ind])
            return boundary_list
        else:
            ROI_dilate = binary_dilation(input=self.ROI[:, :, self.ind], structure=kernel, iterations=1)
            return ROI_dilate - self.ROI[:, :, self.ind]


    def MergeDataWithROI(self, boundary):
        if isinstance(boundary, list):
            imshow_data = np.stack((self.X[:, :, self.ind], self.X[:, :, self.ind], self.X[:, :, self.ind]), axis=2)
            index_x, index_y = np.where(boundary[0] == 1)
            imshow_data[index_x, index_y, :] = 0
            imshow_data[index_x, index_y, 0] = np.max(self.X[:, :, self.ind])
            if len(boundary) > 1:
                index_x, index_y = np.where(boundary[1] == 1)
                imshow_data[index_x, index_y, :] = 0
                imshow_data[index_x, index_y, 1] = np.max(self.X[:, :, self.ind])
            if len(boundary) > 2:
                index_x, index_y = np.where(boundary[2] == 1)
                imshow_data[index_x, index_y, :] = 0
                imshow_data[index_x, index_y, 2] = np.max(self.X[:, :, self.ind])
        else:
            imshow_data = np.stack((self.X[:, :, self.ind], self.X[:, :, self.ind], self.X[:, :, self.ind]), axis=2)
            index_x, index_y = np.where(boundary == 1)
            imshow_data[index_x, index_y, :] = 0
            imshow_data[index_x, index_y, 0] = np.max(self.X[:, :, self.ind])
        return imshow_data

    def update(self):
        if isinstance(self.ROI, list):
            imshow_data = self.MergeDataWithROI(self.FindBoundaryOfROI())
            self.im.set_data(imshow_data)
        elif np.max(self.ROI) == 0:
            self.im.set_data(self.X[:, :, self.ind])
        else:
            imshow_data = self.MergeDataWithROI(self.FindBoundaryOfROI())
            self.im.set_data(imshow_data)

        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        # if self.ROI.any() == None:
        #     print('There is no ROI')
        # else:
        #     self.im = ax.contour(self.ROI[:, :, self.ind], colors='y')

def Imshow3D(data, vmin=None, vmax=None, ROI=0, name=' '):
    fig, ax = plt.subplots(1, 1, )
    tracker = IndexTracker(ax, data, vmin, vmax, ROI)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.set_window_title(name)
    plt.show()

################################################

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    ax.set_title('slice %d' % ax.index)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[..., ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[..., ax.index])

def Imshow3DConsole(volume, vmin=None, vmax=None):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[2] // 2
    ax.imshow(volume[..., ax.index], cmap='Greys_r', vmin=vmin, vmax=vmax)
    fig.canvas.mpl_connect('key_press_event', process_key)

##############################################################
def FlattenAllSlices(data):
    assert(data.ndim == 3)
    row, col, slice = data.shape
    width = 1
    while True:
        if width * width >= slice:
            break
        else:
            width += 1
    imshow_data = np.zeros((row * width, col * width))
    slice_indexs = range(0, slice)
    x, y = Index2XY(slice_indexs, (width, width))

    for x_index, y_index, slice_index in zip(x, y, slice_indexs):
        imshow_data[x_index * row : (x_index + 1) * row, y_index * row : (y_index + 1) * row] = data[..., slice_index]

    plt.imshow(Normalize01(imshow_data), cmap='gray')
    plt.show()

################################################################
# 该函数将每个2d图像进行变换。
def MergeImageWithROI(data, roi):
    if data.ndim >= 3:
        print("Should input 2d image")
        return data

    if not isinstance(roi, list):
        roi = [roi]

    if len(roi) > 3:
        print('Only show 3 ROIs')
        return data

    data = np.asarray(Normalize01(data) * 255, dtype=np.uint8)

    kernel = np.ones((3, 3))
    new_data = np.stack([data, data, data], axis=2)
    boundary = binary_dilation(input=roi[0], structure=kernel, iterations=1) - roi[0]
    index_x, index_y = np.where(boundary == 1)
    new_data[index_x, index_y, :] = 0
    new_data[index_x, index_y, 0] = np.max(data)
    if len(roi) > 1:
        boundary = binary_dilation(input=roi[1], structure=kernel, iterations=1) - roi[1]
        index_x, index_y = np.where(boundary == 1)
        new_data[index_x, index_y, :] = 0
        new_data[index_x, index_y, 1] = np.max(data)
    if len(roi) > 2:
        boundary = binary_dilation(input=roi[2], structure=kernel, iterations=1) - roi[2]
        index_x, index_y = np.where(boundary == 1)
        new_data[index_x, index_y, :] = 0
        new_data[index_x, index_y, 2] = np.max(data)
    return new_data

def FusionImage(data, mask, is_show=False):
    '''
    To Fusion two 2D images.
    :param data: The background
    :param mask: The fore-ground
    :param is_show: Boolen. If set to True, to show the result; else to return the fusion image. (RGB).
    :return:
    '''
    if data.ndim >= 3:
        print("Should input 2d image")
        return data

    dpi = 96
    x, y = data.shape
    w = y / dpi
    h = x / dpi

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(data, cmap='gray')
    plt.imshow(mask, cmap='hot', alpha=0.3)

    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)

    if is_show:
        plt.show()
    else:
        plt.axis('off')
        plt.savefig('temp.jpg', format='jpeg', aspect='normal', bbox_inches='tight', pad_inches=0.0)
        array = imread('temp.jpg')
        os.remove('temp.jpg')
        return array


def ShowColorByROI(array, roi, color_map='jet', store_path='', is_show=True):
    if array.shape != roi.shape:
        print('Array and ROI must have same shape')
        return

    array = Normalize01(array)
    cmap = plt.get_cmap(color_map)
    rgba_array = cmap(array)
    rgb_array = np.delete(rgba_array, 3, 2)

    print(array.shape)
    print(rgb_array.shape)

    index_roi_x, index_roi_y = np.where(roi == 0)
    for index_x, index_y in zip(index_roi_x, index_roi_y):
        rgb_array[index_x, index_y, :] = array[index_x, index_y]

    plt.imshow(rgb_array)
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if store_path:
        plt.savefig(store_path, format='tif', dpi=300, bbox_inches='tight', pad_inches=0)
    if is_show:
        plt.show()
    plt.close()