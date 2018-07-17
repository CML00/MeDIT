import numpy as np
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
from scipy import ndimage
from MeDIT.ImageProcess import RemoveSmallRegion, XY2Index, XYZ2Index

def AUC_Confidence_Interval(y_true, y_pred, CI_index=0.95):
    AUC = roc_auc_score(y_true, y_pred)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int((1.0 - CI_index)/2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index)/2 * len(sorted_scores))]
    CI = [confidence_lower, confidence_upper]

    print('AUC is {:.3f}, Confidence interval : [{:0.3f} - {:0.3}]'.format(AUC, confidence_lower, confidence_upper))
    return AUC, CI, sorted_scores

def StatisticDetection(prediction_map, label_map, threshold_value_on_overlap=0.5):
    '''
    To Calculate the staitstic value of the detection results.
    For each region of the label map, if the prediction overlaps above threshold value, true positive value + 1.
    For each region of the label map, if the prediction overlaps under threshold value, false negative value + 1.
    For each region of the prediction map, if the label overlaps under threshold value, false positive value + 1.

    :param prediction_map: the detection result (binary image)
    :param label_map: the ground truth (binary image)
    :param threshold_value_on_overlap: the threshold value to estiamte the reuslt.
    :return: the true positive value, false positive value, and the false negative value.
    '''
    image_shape = prediction_map.shape

    prediction_im, prediction_nb = ndimage.label(prediction_map)
    label_im, label_nb = ndimage.label(label_map)

    true_positive = 0
    false_negative = 0
    false_positive = 0

    for index in range(1, label_nb+1):
        x_label, y_label = np.where(label_im == index)
        index_label = XY2Index([x_label, y_label], image_shape)

        x_pred, y_pred = np.where(prediction_im > 0)
        index_pred = XY2Index([x_pred, y_pred], image_shape)
        inter_index = np.intersect1d(index_label, index_pred)

        if inter_index.size / index_label.size >= threshold_value_on_overlap:
            true_positive += 1
        else:
            false_negative += 1

    for index in range(1, prediction_nb+1):
        x_pred, y_pred = np.where(prediction_im == index)
        index_pred = XY2Index([x_pred, y_pred], image_shape)

        x_label, y_label = np.where(label_im > 0)
        index_label = XY2Index([x_label, y_label], image_shape)
        inter_index = np.intersect1d(index_label, index_pred)

        if inter_index.size / index_pred.size < threshold_value_on_overlap:
            false_positive += 1

    return true_positive, false_positive, false_negative

def StatsticOverlap(prediction_map, label_map):
    image_shape = prediction_map.shape

    if prediction_map.ndim == 2:
        x_label, y_label = np.where(label_map > 0)
        index_label = XY2Index([x_label, y_label], image_shape)

        x_pred, y_pred = np.where(prediction_map > 0)
        index_pred = XY2Index([x_pred, y_pred], image_shape)
    elif prediction_map.ndim == 3:
        x_label, y_label, z_label = np.where(label_map > 0)
        index_label = XYZ2Index([x_label, y_label, z_label], image_shape)

        x_pred, y_pred, z_pred = np.where(prediction_map > 0)
        index_pred = XYZ2Index([x_pred, y_pred, z_pred], image_shape)

    inter_index = np.intersect1d(index_label, index_pred)

    true_positive_value = len(inter_index)
    false_positive_value = len(index_pred) - len(inter_index)
    false_negative_value = len(index_label) - len(inter_index)

    return true_positive_value, false_positive_value, false_negative_value