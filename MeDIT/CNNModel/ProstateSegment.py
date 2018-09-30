import numpy as np
import os
from scipy import ndimage as nd
from configparser import ConfigParser
from scipy import ndimage
from keras.models import model_from_yaml

from MeDIT.CNNModel.ImagePrepare import ImagePrepare
from MeDIT.SaveAndLoad import GetDataFromSimpleITK
from MeDIT.Normalize import Normalize
from MeDIT.SaveAndLoad import GetImageFromArray, SaveNiiImage
from MeDIT.ArrayProcess import Crop3DImage

class ProstateSegmentation2D:
    def __init__(self):
        self._loaded_model = None
        self._selected_slice_index = None
        self._raw_data_shape = None
        self._selected_index = dict()
        self._image_preparer = ImagePrepare()

    def __RemoveSmallRegion(self, mask, size_thres=2000):
        # seperate each connected ROI
        label_im, nb_labels = ndimage.label(mask)

        # remove small ROI
        for i in range(1, nb_labels + 1):
            if (label_im == i).sum() < size_thres:
                # remove the small ROI in mask
                mask[label_im == i] = 0
        return mask

    def __KeepLargest(self, mask):
        label_im, nb_labels = ndimage.label(mask)
        max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
        index = np.argmax(max_volume)
        new_mask = np.zeros(mask.shape)
        new_mask[label_im == index + 1] = 1
        return new_mask

    def LoadModel(self, fold_path):

        model_path = os.path.join(fold_path, 'model.yaml')

        if not self.CheckFileExistence(model_path): return

        yaml_file = open(model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self._loaded_model = model_from_yaml(loaded_model_yaml)

        weight_path = os.path.join(fold_path, 'weights.h5')
        if self.CheckFileExistence(weight_path):
            self._loaded_model.load_weights(weight_path)

    def CheckFileExistence(self, filepath):
        if os.path.isfile(filepath):
            return True
        else:
            print('Current file not exist or path is not correct')
            print('current filepath:' + os.path.abspath('.') + '\\' + filepath)
            return False

    def TransDataFor2DModel(self, data):
        data = data.swapaxes(0, 2).swapaxes(1, 2)  # Exchange the order of axis
        data = data[..., np.newaxis]  # Add channel axis
        return data

    def invTransDataFor2DModel(self, preds):
        preds = np.squeeze(preds)
        # Exchange the order of axis back
        preds = preds.swapaxes(1, 2).swapaxes(0, 2)
        return preds

    def Run(self, image, model_folder_path, store_folder=''):
        resolution = image.GetSpacing()
        _, data = GetDataFromSimpleITK(image, dtype=np.float32)

        self._image_preparer.LoadModelConfig(os.path.join(model_folder_path, 'config.ini'))

        ''' 2) Select Data'''

        data = self._image_preparer.CropDataShape(data, resolution)
        data = self.TransDataFor2DModel(data)
        data = Normalize(data)

        preds = self._loaded_model.predict(data)

        preds = preds[:, -np.prod(self._image_preparer.GetShape()):, :]
        preds = np.reshape(preds, (
        data.shape[0], self._image_preparer.GetShape()[0], self._image_preparer.GetShape()[1]))

        # ct the ROI
        preds = self.invTransDataFor2DModel(preds)
        preds = self._image_preparer.RecoverDataShape(preds, resolution)

        mask = np.asarray(preds > 0.5, dtype=np.uint8)
        mask = self.__KeepLargest(mask)

        # To process the extremely cases
        final_shape = image.GetSize()
        final_shape = [final_shape[1], final_shape[0], final_shape[2]]
        mask = Crop3DImage(mask, final_shape)

        mask_image =  GetImageFromArray(mask, image)
        if store_folder:
            if os.path.isdir(store_folder):
                store_folder = os.path.join(store_folder, 'prostate_roi.nii.gz')
            SaveNiiImage(store_folder, mask_image)

        return mask_image, mask


def testSeg():
    model_folder_path = r'C:\MyCode\MPApp\DPmodel\ProstateSegmentation'
    prostate_segmentor = ProstateSegmentation2D()
    prostate_segmentor.LoadModel(model_folder_path)

    from MeDIT.SaveAndLoad import LoadNiiData
    image, _, show_data = LoadNiiData(r'C:\Users\SY\Desktop\1.2.156.14702.6.146.20150329000062\501_t2_tse_tra.nii', dtype=np.float32, is_show_info=True)

    predict_image, prodict_data = prostate_segmentor.Run(image, model_folder_path)

    from MeDIT.Visualization import Imshow3D
    from MeDIT.Normalize import Normalize01
    Imshow3D(Normalize01(show_data), ROI=np.asarray(prodict_data > 0.5, dtype=np.uint8))


if __name__ == '__main__':
    testSeg()
