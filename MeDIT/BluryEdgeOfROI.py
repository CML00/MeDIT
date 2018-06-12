from scipy.ndimage.morphology import binary_dilation, binary_erosion
import numpy as np

def BluryEdgeOfROI(initial_ROI):
    if len(np.shape(initial_ROI)) == 2:
        kernel = np.ones((3, 3))
    elif len(np.shape(initial_ROI)) == 3:
        kernel = np.ones((3, 3, 3))
    else:
        print('Only could process 2D or 3D data')
        return []
        
    
    initial_ROI = initial_ROI == 1
    
    ROI_dilate = binary_dilation(input=initial_ROI, structure=kernel, iterations=1)
    ROI_erode = binary_erosion(input=ROI_dilate, structure=kernel, iterations=1)
    ROI_erode1 = binary_erosion(input=ROI_erode, structure=kernel, iterations=1)
    ROI_erode2 = binary_erosion(input=ROI_erode1, structure=kernel, iterations=1)
    
    dif_dilate = (ROI_dilate - ROI_erode) * 0.25
    dif_ori = (ROI_erode - ROI_erode1) * 0.5
    dif_in = (ROI_erode1 - ROI_erode2) * 0.75
    
    blurred_ROI = ROI_erode2 + dif_dilate + dif_ori + dif_in

    return blurred_ROI