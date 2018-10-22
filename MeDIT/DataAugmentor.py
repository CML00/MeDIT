import numpy as np
from scipy.interpolate import RegularGridInterpolator
from collections import OrderedDict

class AugmentParametersGenerator():
    '''
    To generate random parameters for 2D or 3D numpy array transform.
    '''
    def __init__(self):
        self.stretch_x = 0.0
        self.stretch_y = 0.0
        self.stretch_z = 0.0
        self.shear = 0.0
        self.rotate_x_angle = 0.0
        self.rotate_z_angle = 0.0
        self.shift_x = 0
        self.shift_y = 0
        self.shift_z = 0
        self.horizontal_flip = False
        self.vertical_flip = False
        self.slice_flip = False

    def RandomParameters(self, parameter_dict):
        # Stretch
        if 'stretch_x' in parameter_dict:
            self.stretch_x = np.random.randint((1 - parameter_dict['stretch_x']) * 100,
                                                                   (1 + parameter_dict['stretch_x']) * 100, 1) / 100
        else:
            self.stretch_x = 1.0

        if 'stretch_y' in parameter_dict:
            self.stretch_y = np.random.randint((1 - parameter_dict['stretch_y']) * 100,
                                                                   (1 + parameter_dict['stretch_y']) * 100, 1) / 100
        else:
            self.stretch_y = 1.0

        if 'stretch_z' in parameter_dict:
            self.stretch_z = np.random.randint((1 - parameter_dict['stretch_z']) * 100,
                                                                   (1 + parameter_dict['stretch_z']) * 100, 1) / 100
        else:
            self.stretch_z = 1.0

        # Shear and rotate
        if 'shear' in parameter_dict:
            self.shear = np.random.randint(-parameter_dict['shear'] * 100,
                                                               parameter_dict['shear'] * 100, 1) / 100
        else:
            self.shear = 0.0

        if 'rotate_z_angle' in parameter_dict:
            self.rotate_z_angle = np.random.randint(-parameter_dict['rotate_z_angle'] * 100,
                                                                        parameter_dict['rotate_z_angle'] * 100, 1) / 100
        else:
            self.rotate_z_angle = 0.0

        if 'rotate_x_angle' in parameter_dict:
            self.rotate_x_angle = np.random.randint(-parameter_dict['rotate_x_angle'] * 100,
                                                                        parameter_dict['rotate_x_angle'] * 100, 1) / 100
        else:
            self.rotate_x_angle = 0.0

        # Shift
        if 'shift_x' in parameter_dict:
            if parameter_dict['shift_x'] < 0.9999:
                self.shift_x = parameter_dict['shift_x'] * (np.random.random((1,)) - 0.5)
            else:
                self.shift_x = \
                np.random.randint(-parameter_dict['shift_x'], parameter_dict['shift_x'], (1,))[0]
        else:
            self.shift_x = 0

        if 'shift_y' in parameter_dict:
            if parameter_dict['shift_y'] < 0.9999:
                self.shift_y = parameter_dict['shift_y'] * (np.random.random((1,)) - 0.5)
            else:
                self.shift_y = \
                np.random.randint(-parameter_dict['shift_y'], parameter_dict['shift_y'], (1,))[0]
        else:
            self.shift_y = 0

        if 'shift_z' in parameter_dict:
            if parameter_dict['shift_z'] < 0.9999:
                self.shift_z = parameter_dict['shift_z'] * (np.random.random((1,)) - 0.5)
            else:
                self.shift_z = \
                np.random.randint(-parameter_dict['shift_z'], parameter_dict['shift_z'], (1,))[0]
        else:
            self.shift_z = 0

        # Flip
        if ('horizontal_flip' in parameter_dict) and (parameter_dict['horizontal_flip']):
            self.horizontal_flip = np.random.choice([True, False])
        else:
            self.horizontal_flip= False
        if ('vertical_flip' in parameter_dict) and (parameter_dict['vertical_flip']):
            self.vertical_flip = np.random.choice([True, False])
        else:
            self.vertical_flip= False
        if ('slice_flip' in parameter_dict) and (parameter_dict['slice_flip']):
            self.slice_flip = np.random.choice([True, False])
        else:
            self.slice_flip = False

    def GetRandomParametersDict(self):

        return OrderedDict(sorted(self.__dict__.items(), key=lambda t: t[0]))

class DataAugmentor3D():
    '''
    To process 3D numpy array transform. The transform contains: stretch in 3 dimensions, shear along x direction,
    rotation around z and x axis, shift along x, y, z direction, and flip along x, y, z direction.
    '''
    def __init__(self):
        self.stretch_x = 1.0
        self.stretch_y = 1.0
        self.stretch_z = 1.0
        self.shear = 0.0
        self.rotate_x_angle = 0.0
        self.rotate_z_angle = 0.0
        self.shift_x = 0
        self.shift_y = 0
        self.shift_z = 0
        self.horizontal_flip = False
        self.vertical_flip = False
        self.slice_flip = False
        
    def SetParameter(self, parameter_dict):
        if 'stretch_x' in parameter_dict: self.stretch_x = parameter_dict['stretch_x']
        if 'stretch_y' in parameter_dict: self.stretch_y = parameter_dict['stretch_y']
        if 'stretch_z' in parameter_dict: self.stretch_z = parameter_dict['stretch_z']
        if 'shear' in parameter_dict: self.shear = parameter_dict['shear']
        if 'rotate_z_angle' in parameter_dict: self.rotate_z_angle = parameter_dict['rotate_z_angle']
        if 'rotate_x_angle' in parameter_dict: self.rotate_x_angle = parameter_dict['rotate_x_angle']
        if 'shift_x' in parameter_dict: self.shift_x = parameter_dict['shift_x']
        if 'shift_y' in parameter_dict: self.shift_y = parameter_dict['shift_y']
        if 'shift_z' in parameter_dict: self.shift_z = parameter_dict['shift_z']
        if 'horizontal_flip' in parameter_dict: self.horizontal_flip = parameter_dict['horizontal_flip']
        if 'vertical_flip' in parameter_dict: self.vertical_flip = parameter_dict['vertical_flip']
        if 'slice_flip' in parameter_dict: self.slice_flip = parameter_dict['slice_flip']

    def GetTransformMatrix3D(self):
        transform_matrix = np.zeros((3, 3))
        transform_matrix[0, 0] = self.stretch_x
        transform_matrix[1, 1] = self.stretch_y
        transform_matrix[2, 2] = self.stretch_z
        transform_matrix[1, 0] = self.shear

        rotate_x_angle = self.rotate_x_angle / 180.0 * np.pi
        rotate_z_angle = self.rotate_z_angle / 180.0 * np.pi

        rotate_x_matrix = [[1, 0, 0],
                           [0, np.cos(rotate_x_angle), -np.sin(rotate_x_angle)],
                           [0, np.sin(rotate_x_angle), np.cos(rotate_x_angle)]]
        rotate_z_matrix = [[np.cos(rotate_z_angle), -np.sin(rotate_z_angle), 0],
                           [np.sin(rotate_z_angle), np.cos(rotate_z_angle), 0],
                           [0, 0, 1]]

        return transform_matrix.dot(np.dot(rotate_x_matrix, rotate_z_matrix))

    def Shift3DImage(self, data):
        non = lambda s: s if s < 0 else None
        mom = lambda s: max(0, s)

        shifted_data = np.zeros_like(data)
        shifted_data[mom(self.shift_x):non(self.shift_x), mom(self.shift_y):non(self.shift_y), mom(self.shift_z):non(self.shift_z)] = \
            data[mom(-self.shift_x):non(-self.shift_x), mom(-self.shift_y):non(-self.shift_y), mom(-self.shift_z):non(-self.shift_z)]
        return shifted_data

    def Flip3DImage(self, data):
        if self.horizontal_flip: data = np.flip(data, axis=1)
        if self.vertical_flip: data = np.flip(data, axis=0)
        if self.slice_flip: data = np.flip(data, axis=2)
        return data

    def ClearParameters(self):
        self.__init__()

    def Transform3Ddata(self, data, transform_matrix, method='nearest'):
        new_data = np.copy(np.float32(data))

        image_row, image_col, image_slice = np.shape(new_data)
        x_vec = np.array(range(image_row)) - image_row // 2
        y_vec = np.array(range(image_col)) - image_col // 2
        z_vec = np.array(range(image_slice)) - image_slice // 2

        x_raw, y_raw, z_raw = np.meshgrid(x_vec, y_vec, z_vec)
        x_raw = x_raw.flatten()
        y_raw = y_raw.flatten()
        z_raw = z_raw.flatten()
        point_raw = np.transpose(np.stack([x_raw, y_raw, z_raw], axis=1))

        points = np.transpose(transform_matrix.dot(point_raw))

        my_interpolating_function = RegularGridInterpolator((x_vec, y_vec, z_vec), new_data, method=method,
                                                            bounds_error=False, fill_value=0)
        temp = my_interpolating_function(points)
        result = np.reshape(temp, (image_row, image_col, image_slice))
        result = np.transpose(result, (1, 0, 2))  # 经过插值之后, 行列的顺序会颠倒
        return result

    def Execute(self, source_data, aug_parameter={}, interpolation_method='nearest', is_clear=False):
        if source_data.ndim != 3:
            print('Input the data with 3 dimensions!')
            return source_data
        if aug_parameter != {}:
            self.SetParameter(aug_parameter)

        # from Imshow3D import FlattenAllSlices
        # from Normalize import Normalize01

        transform_matrix = self.GetTransformMatrix3D()
        # print(transform_matrix)
        target_data = self.Transform3Ddata(source_data, transform_matrix=transform_matrix, method=interpolation_method)
        # FlattenAllSlices(Normalize01(source_data))
        # FlattenAllSlices(Normalize01(target_data))

        target_data = self.Shift3DImage(target_data)
        # FlattenAllSlices(Normalize01(target_data))
        target_data = self.Flip3DImage(target_data)
        # FlattenAllSlices(Normalize01(target_data))

        if is_clear:
            self.ClearParameters()

        return target_data

class DataAugmentor2D():
    '''
    To process 2D numpy array transform. The transform contains: stretch in x, y dimensions, shear along x direction,
    rotation, shift along x, y direction, and flip along x, y direction.
    '''
    def __init__(self):
        self.stretch_x = 1.0
        self.stretch_y = 1.0
        self.shear = 0.0
        self.rotate_z_angle = 0.0
        self.shift_x = 0
        self.shift_y = 0
        self.horizontal_flip = False
        self.vertical_flip = False

    def SetParameter(self, parameter_dict):
        if 'stretch_x' in parameter_dict: self.stretch_x = parameter_dict['stretch_x']
        if 'stretch_y' in parameter_dict: self.stretch_y = parameter_dict['stretch_y']
        if 'shear' in parameter_dict: self.shear = parameter_dict['shear']
        if 'rotate_z_angle' in parameter_dict: self.rotate_z_angle = parameter_dict['rotate_z_angle']
        if 'shift_x' in parameter_dict: self.shift_x = parameter_dict['shift_x']
        if 'shift_y' in parameter_dict: self.shift_y = parameter_dict['shift_y']
        if 'horizontal_flip' in parameter_dict: self.horizontal_flip = parameter_dict['horizontal_flip']
        if 'vertical_flip' in parameter_dict: self.vertical_flip = parameter_dict['vertical_flip']

    def GetTransformMatrix2D(self):
        transform_matrix = np.zeros((2, 2))
        transform_matrix[0, 0] = self.stretch_x
        transform_matrix[1, 1] = self.stretch_y
        transform_matrix[1, 0] = self.shear

        rotate_z_angle = self.rotate_z_angle / 180.0 * np.pi

        rotate_z_matrix = np.squeeze(np.asarray([[np.cos(rotate_z_angle), -np.sin(rotate_z_angle)],
                           [np.sin(rotate_z_angle), np.cos(rotate_z_angle)]]))

        return transform_matrix.dot(rotate_z_matrix)

    def Shift2DImage(self, data):
        non = lambda s: s if s < 0 else None
        mom = lambda s: max(0, s)

        shifted_data = np.zeros_like(data)
        shifted_data[mom(self.shift_x):non(self.shift_x), mom(self.shift_y):non(self.shift_y),] = \
            data[mom(-self.shift_x):non(-self.shift_x), mom(-self.shift_y):non(-self.shift_y)]
        return shifted_data

    def Flip2DImage(self, data):
        if self.horizontal_flip: data = np.flip(data, axis=1)
        if self.vertical_flip: data = np.flip(data, axis=0)
        return data

    def ClearParameters(self):
        self.__init__()

    def Transform2Ddata(self, data, transform_matrix, method='nearest'):
        new_data = np.copy(np.float32(data))

        image_row, image_col = np.shape(new_data)
        x_vec = np.array(range(image_row)) - image_row // 2
        y_vec = np.array(range(image_col)) - image_col // 2

        x_raw, y_raw = np.meshgrid(x_vec, y_vec)
        x_raw = x_raw.flatten()
        y_raw = y_raw.flatten()
        point_raw = np.transpose(np.stack([x_raw, y_raw], axis=1))

        points = np.transpose(transform_matrix.dot(point_raw))

        my_interpolating_function = RegularGridInterpolator((x_vec, y_vec), new_data, method=method,
                                                            bounds_error=False, fill_value=0)
        temp = my_interpolating_function(points)
        result = np.reshape(temp, (image_row, image_col))
        result = np.transpose(result, (1, 0))  # 经过插值之后, 行列的顺序会颠倒
        return result

    def Execute(self, source_data, aug_parameter={}, interpolation_method='nearest', is_clear=False):
        if source_data.ndim != 2:
            print('Input the data with 2 dimensions!')
            return source_data
        if aug_parameter != {}:
            self.SetParameter(aug_parameter)

        transform_matrix = self.GetTransformMatrix2D()
        target_data = self.Transform2Ddata(source_data, transform_matrix=transform_matrix, method=interpolation_method)

        target_data = self.Shift2DImage(target_data)
        target_data = self.Flip2DImage(target_data)

        if is_clear:
            self.ClearParameters()

        return target_data

def main():
    pass
    # random_params = {'stretch_x': 0.1, 'stretch_y': 0.1, 'shear': 0.1, 'rotate_z_angle': 20, 'horizontal_flip': True}
    # param_generator = AugmentParametersGenerator()
    # aug_generator = DataAugmentor2D()

    # from MeDIT.SaveAndLoad import LoadNiiData
    # _, _, data = LoadNiiData(data_path)
    # _, _, roi = LoadNiiData(roi_path)

    # data = data[..., data.shape[2] // 2]
    # roi = roi[..., roi.shape[2] // 2]

    # from Visualization import DrawBoundaryOfBinaryMask
    # from Normalize import Normalize01

    # while True:
    #     param_generator.RandomParameters(random_params)
    #     aug_generator.SetParameter(param_generator.GetRandomParametersDict())
    #
    #     new_data = aug_generator.Execute(data, interpolation_method='linear')
    #     new_roi = aug_generator.Execute(roi, interpolation_method='nearest')
    #     DrawBoundaryOfBinaryMask(Normalize01(new_data), new_roi)

def Test2D():
    import numpy as np
    data = np.reshape(np.arange(144), (12, 12))
    data_augmentor = DataAugmentor2D()
    result = data_augmentor.Execute(data, aug_parameter={'stretch_x': 0.5}, interpolation_method='linear')
    print(result)

if __name__ == '__main__':
    Test2D()
