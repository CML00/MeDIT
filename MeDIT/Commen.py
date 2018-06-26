import SimpleITK as sitk
import numpy as np

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


