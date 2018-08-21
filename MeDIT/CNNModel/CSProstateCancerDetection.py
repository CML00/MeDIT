import numpy as np
import os
from configparser import ConfigParser
import SimpleITK as sitk
from keras.models import model_from_yaml

from MeDIT.ImageProcess import ExtractPatch, XY2Index
from MeDIT.CNNModel.ImagePrepare import ImagePrepare
from MeDIT.CNNModel.ProstateSegment import ProstateSegmentation2D
from MeDIT.Normalize import NormalizeForModality, NormalizeByROI
from MeDIT.SaveAndLoad import GetDataFromSimpleITK, GetImageFromArray, SaveNiiImage
from scipy import ndimage
from scipy.ndimage import filters


class CST2AdcDwiDetect():
    def __init__(self):
        self.__model = None
        self.__image_preparer = ImagePrepare()

    def __RemoveSmallRegion(self, mask, size_thres=50):
        # seperate each connected ROI
        label_im, nb_labels = ndimage.label(mask)

        # remove small ROI
        for i in range(1, nb_labels + 1):
            if (label_im == i).sum() < size_thres:
                # remove the small ROI in mask
                mask[label_im == i] = 0
        return mask

    def LoadModel(self, fold_path):
        from keras.models import model_from_yaml
        model_path = os.path.join(fold_path, 'model.yaml')

        if not self.CheckFileExistence(model_path): return

        yaml_file = open(model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.__model = model_from_yaml(loaded_model_yaml)

        weight_path = os.path.join(fold_path, 'weights.h5')
        if self.CheckFileExistence(weight_path):
            self.__model.load_weights(weight_path)

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

    # Do Detect
    def Run(self, t2_image, adc_image, dwi_image, detect_model_path, seg_model_path='', prostate_roi_image=None, store_folder=''):
        if len(seg_model_path) != 0 :
            prostate_segmentor = ProstateSegmentation2D()
            prostate_segmentor.LoadModel(seg_model_path)
            prostate_roi_image, prostate_roi = prostate_segmentor.Run(t2_image, seg_model_path)
        else:
            _, prostate_roi = GetDataFromSimpleITK(prostate_roi_image, dtype=np.uint8)

        self.__image_preparer.LoadModelConfig(os.path.join(detect_model_path, 'config.ini'))
        if self.__model == None:
            self.LoadModel(fold_path=detect_model_path)
            
        if not ((t2_image.GetSpacing() == adc_image.GetSpacing()) and (dwi_image.GetSpacing() == prostate_roi_image.GetSpacing())):
            print("The spacing is not consistant among mp-mr images")
            return
        if not ((t2_image.GetSize() == adc_image.GetSize()) and (dwi_image.GetSize() == prostate_roi_image.GetSize())):
            print("The size is not consistant among mp-mr images")
            return

        _, t2 = GetDataFromSimpleITK(t2_image)
        _, adc = GetDataFromSimpleITK(adc_image)
        _, dwi = GetDataFromSimpleITK(dwi_image)

        resolution = t2_image.GetSpacing()

        pred = np.zeros(t2.shape)
        for slice_index in range(t2.shape[-1]):
            t2_slice = np.asarray(t2[..., slice_index], dtype=np.float32)
            adc_slice = np.asarray(adc[..., slice_index], dtype=np.float32)
            dwi_slice = np.asarray(dwi[..., slice_index], dtype=np.float32)
            prostate_roi_slice = prostate_roi[..., slice_index]

            roi_index_x, roi_index_y = np.where(prostate_roi_slice > 0.5)
            if roi_index_x.size < 10 or roi_index_y.size < 10:
                continue
            else:
                center_x = (np.max(roi_index_x) + np.min(roi_index_x)) // 2
                center_y = (np.max(roi_index_y) + np.min(roi_index_y)) // 2

            if np.std(adc_slice) < 1e-4 or np.std(dwi_slice) < 1e-4:
                continue

            t2_slice = self.__image_preparer.CropDataShape(t2_slice, resolution, [center_x, center_y])
            adc_slice = self.__image_preparer.CropDataShape(adc_slice, resolution, [center_x, center_y])
            dwi_slice = self.__image_preparer.CropDataShape(dwi_slice, resolution, [center_x, center_y])
            prostate_roi_slice = self.__image_preparer.CropDataShape(prostate_roi_slice, resolution,
                                                                     [center_x, center_y])

            t2_slice = NormalizeByROI(t2_slice, np.asarray(prostate_roi_slice > 0.5, dtype=np.uint8))
            adc_slice = NormalizeByROI(adc_slice, np.asarray(prostate_roi_slice > 0.5, dtype=np.uint8))
            dwi_slice = NormalizeByROI(dwi_slice, np.asarray(prostate_roi_slice > 0.5, dtype=np.uint8))

            t2_slice = t2_slice[np.newaxis, ..., np.newaxis]
            adc_slice = adc_slice[np.newaxis, ..., np.newaxis]
            dwi_slice = dwi_slice[np.newaxis, ..., np.newaxis]

            input_list = [t2_slice, adc_slice, dwi_slice]

            pred_slice = self.__model.predict(input_list)
            pred_slice = np.squeeze(pred_slice)
            pred_slice = pred_slice[..., np.newaxis]

            pred[..., slice_index] = np.squeeze(self.__image_preparer.RecoverDataShape(pred_slice, resolution))

        mask = np.asarray(pred > 0.5, dtype=np.uint8)
        mask = self.__RemoveSmallRegion(mask, 20)

        mask_image = GetImageFromArray(mask, t2_image)
        pred_image = GetImageFromArray(pred, t2_image)
        if store_folder:
            if os.path.isdir(store_folder):
                roi_output = os.path.join(store_folder, 'CS_PCa_ROI.nii')
                SaveNiiImage(roi_output, mask_image)
                pred_output = os.path.join(store_folder, 'CS_PCa_Pred.nii')
                SaveNiiImage(pred_output, pred_image)

        return mask_image, mask

def testDetect():
    model_folder_path = r'C:\MyCode\MPApp\DPmodel\CSPCaDetection'
    pca_detect = CST2AdcDwiDetect()
    pca_detect.LoadModel(model_folder_path)

    from MeDIT.SaveAndLoad import LoadNiiData
    t2_image, _, t2_data = LoadNiiData(r'c:\MyCode\MPApp\ProstateX-0004\005_t2_tse_tra.nii', dtype=np.float32, is_show_info=True)
    _, _, dwi_data = LoadNiiData(r'c:\MyCode\MPApp\ProstateX-0004\006_ep2d_diff_tra_DYNDIST_MIX_Reg.nii', dtype=np.float32, is_show_info=True)
    adc_image, _, adc_data = LoadNiiData(r'c:\MyCode\MPApp\ProstateX-0004\007_ep2d_diff_tra_DYNDIST_MIX_ADC_Reg.nii', dtype=np.float32, is_show_info=True)

    dwi_data = dwi_data[..., -1]
    dwi_image = GetImageFromArray(dwi_data, adc_image)

    print(t2_data.shape, adc_data.shape, dwi_data.shape)

    predict_image, prodict_data = pca_detect.Run(t2_image, adc_image, dwi_image,
                                                 model_folder_path, seg_model_path=r'c:\Users\SY\Desktop\model\StoreModel\ProstateSegmentation')

    from MeDIT.Visualization import Imshow3D
    from MeDIT.Normalize import Normalize01
    Imshow3D(Normalize01(adc_data), ROI=np.asarray(prodict_data > 0.5, dtype=np.uint8))


if __name__ == '__main__':
    testDetect()