import SimpleITK as sitk
import numpy as np
import math
import sys
import os

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

def GetPhysicaladdress():
    '''
    @summary: return the MAC address of the computer
    '''

    mac = None
    if sys.platform == "win32":
        for line in os.popen("ipconfig /all"):
            # print line
            if line.lstrip().startswith("Physical Address") or line.lstrip().startswith("物理地址"):
                mac = line.split(":")[1].strip().replace("-", ":")
                break

    else:
        for line in os.popen("/sbin/ifconfig"):
            if 'Ether' in line:
                mac = line.split()[4]
                break
    return mac

if __name__ == '__main__':
    # array = np.array([1, 'z', 2.5, 1e-4, np.nan, '3'])
    # for index in np.arange(array.size):
    #     print(IsValidNumber(array[index]))
    print(GetPhysicaladdress())
    
