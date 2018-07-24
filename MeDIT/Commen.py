import SimpleITK as sitk
import numpy as np
import math

def IsNumber(string):
    '''
    To adjust the string belongs to a number or not.
    :param string:
    :return:
    '''
    try:
        float(string)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(string)
        return True
    except (TypeError, ValueError):
        pass

    return False

def IsValidNumber(string):
    if not IsNumber(string):
        return False

    if math.isnan(float(string)):
        return False

    return True

if __name__ == '__main__':
    array = np.array([1, 'z', 2.5, 1e-4, np.nan, '3'])
    for index in np.arange(array.size):
        print(IsValidNumber(array[index]))
