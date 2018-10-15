import numpy as np
from sklearn.linear_model import LinearRegression
from MeDIT.Visualization import Imshow3DArray

def MergeKspace(recon_kdata, sampled_kdata, mask, is_fit=True, alpha = 0.5):
    if recon_kdata.shape != mask.shape or sampled_kdata.shape != mask.shape:
        print('Check the shape of these datas')
        return []

    if is_fit:
        fit_x = recon_kdata[mask == 1].flatten()
        fit_y = sampled_kdata[mask == 1].flatten()

        fit_x = fit_x[..., np.newaxis]
        fit_y = fit_y[..., np.newaxis]

        linear_regression = LinearRegression()
        linear_regression.fit(fit_x, fit_y)
        k, b = linear_regression.coef_, linear_regression.intercept_

        recon_kdata = recon_kdata * k + b

    recon_kdata[mask == 1] = alpha * sampled_kdata[mask == 1] + (1 - alpha) * recon_kdata[mask == 1]
    return recon_kdata