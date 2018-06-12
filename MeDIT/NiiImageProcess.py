import dicom2nifti
import pydicom
import os
import shutil
import SimpleITK as sitk
import numpy as np


def GenerateFileName(file_path, name):
    store_path = ''
    if os.path.splitext(file_path)[1] == '.nii':
        store_path = file_path[:-4] + '_' + name + '.nii'
    elif os.path.splitext(file_path)[1] == '.gz':
        store_path = file_path[:-7] + '_' + name + '.nii.gz'
    else:
        print('the input file should be suffix .nii or .nii.gz')

    return store_path

def Dicom2Nii(data_folder, store_folder, store_format='.nii.gz'):
    '''
    Convert Dicom data to Nifty

    :param data_folder: The folder including all DICOM files. Each folder should only contain one serise.
    :param store_folder: The folder that was used to store the nii file. The file name was generated to
    "SeriesNumber + SeriesDescription + .nii.gz"
    :param store_format: The format was used to store. Which should one of '.nii' or '.nii.gz' (default)

    Apr-27-2018, Yang SONG [yang.song.91@foxmail.com]
    '''

    n_files = os.listdir(data_folder)
    try:
        assert(len(n_files) > 0)
    except:
        print("The folder should not be empty!")

    n_files.sort()

    for file in n_files:
        if os.path.splitext(file)[1] != '.dcm' and os.path.splitext(file)[1] != '.IMA':
            print('The fold of {} should only contain dicom files.'.format(data_folder))
            return None

    header = pydicom.read_file(os.path.join(data_folder, n_files[0]))
    file_name = str(header.SeriesNumber).zfill(3) + '_' + header.SeriesDescription + store_format

    file_name = file_name.replace(":", "_")
    file_name = file_name.replace(" ", "_")

    dicom2nifti.dicom_series_to_nifti(data_folder, os.path.join(store_folder, file_name), reorient_nifti=False)

def CommenDicom2Nii(data_folder, store_folder):
    temp_folder = os.path.join(data_folder, 'temp')
    os.mkdir(temp_folder)

    n_files = os.listdir(data_folder)
    n_files.sort()

    for file in n_files:
        if os.path.splitext(file)[1] == '.dcm' or os.path.splitext(file)[1] == '.IMA':
            shutil.move(os.path.join(data_folder, file), os.path.join(temp_folder, file))

    Dicom2Nii(temp_folder, store_folder)

    n_files = os.listdir(temp_folder)
    n_files.sort()
    for file in n_files:
        shutil.move(os.path.join(temp_folder, file), os.path.join(data_folder, file))
    shutil.rmtree(temp_folder)

################################################################################
def ResizeSipmleITKImage(image, expected_resolution=[], expected_shape=[], method=sitk.sitkBSpline, dtype=sitk.sitkFloat32):
    '''
    Resize the SimpleITK image. One of the expected resolution/spacing and final shape should be given.

    :param image: The SimpleITK image.
    :param expected_resolution: The expected resolution.
    :param excepted_shape: The expected final shape.
    :return: The resized image.

    Apr-27-2018, Yang SONG [yang.song.91@foxmail.com]
    '''
    if (expected_resolution == []) and (expected_shape == []):
        print('Give at least one parameters. ')
        return image

    shape = image.GetSize()
    resolution = image.GetSpacing()

    if expected_resolution == []:
        if expected_shape[0] == 0: expected_shape[0] = shape[0]
        if expected_shape[1] == 0: expected_shape[1] = shape[1]
        if expected_shape[2] == 0: expected_shape[2] = shape[2]
        expected_resolution = [raw_resolution * raw_size / dest_size for dest_size, raw_size, raw_resolution in
                               zip(expected_shape, shape, resolution)]
    elif expected_shape == []:
        if expected_resolution[0] == 0: expected_resolution[0] = resolution[0]
        if expected_resolution[1] == 0: expected_resolution[1] = resolution[1]
        if expected_resolution[2] == 0: expected_resolution[2] = resolution[2]
        expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                       dest_resolution, raw_size, raw_resolution in zip(expected_resolution, shape, resolution)]

    output = sitk.Resample(image, expected_shape, sitk.AffineTransform(len(shape)), method, image.GetOrigin(),
                           expected_resolution, image.GetDirection(), dtype)
    return output

def ResizeNiiFile(file_path, store_path='', expected_resolution=[], expected_shape=[], method=sitk.sitkBSpline, dtype=sitk.sitkFloat32):
    if not store_path:
        store_path = GenerateFileName(file_path, 'Resize')

    image = sitk.ReadImage(file_path)
    resized_image = ResizeSipmleITKImage(image, expected_resolution, expected_shape, method=method, dtype=dtype)
    sitk.WriteImage(resized_image, store_path)

################################################################################
def RegistrateImage(fixed_image, moving_image, interpolation_method=sitk.sitkBSpline):
    '''
    Registrate SimpleITK Imageby default parametes.

    :param fixed_image: The reference
    :param moving_image: The moving image.
    :param interpolation_method: The method for interpolation. default is sitkBSpline
    :return: The output image

    Apr-27-2018, Jing ZHANG [798582238@qq.com],
                 Yang SONG [yang.song.91@foxmail.com]
    '''
    if isinstance(fixed_image, str):
        fixed_image = sitk.ReadImage(fixed_image)
    if isinstance(moving_image, str):
        moving_image = sitk.ReadImage(moving_image)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    output_image = sitk.Resample(moving_image, fixed_image, final_transform, interpolation_method, 0.0,
                                     moving_image.GetPixelID())
    return output_image

def RegistrateImageFile(fixed_image_path, moving_image_path, interpolation_method=sitk.sitkBSpline):
    output_image = RegistrateImage(fixed_image_path, moving_image_path, interpolation_method)
    store_path = GenerateFileName(moving_image_path, 'Reg')
    sitk.WriteImage(output_image, store_path)

