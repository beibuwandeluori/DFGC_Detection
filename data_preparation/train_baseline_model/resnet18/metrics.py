import numpy as np
import copy
from sklearn.metrics import roc_auc_score, roc_curve


def get_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc


def get_EER(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER


def get_tn_tp_fn_fp(y_true, y_pred):
    tn = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred))).astype(np.float64)
    tp = np.sum(np.logical_and(y_true, y_pred)).astype(np.float64)
    fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred))).astype(np.float64)
    fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred)).astype(np.float64)
    return tn, tp, fn, fp


def f1_score(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    f1 = 2 * tp / (2 * tp + fp + fn)
    if np.isnan(f1):
        return 0.
    else:
        return f1


def accuracy_score(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if np.isnan(accuracy):
        return 0.
    else:
        return accuracy


def get_metrics(y_true, y_pred):
    try:
        auc = get_auc(y_true, y_pred)
    except:
        auc = 0
    try:
        EER = get_EER(y_true, y_pred)
    except:
        EER = 1.0
    all_f1, all_mcc, all_iou = [], [], []
    y_pred_tmp = copy.deepcopy(y_pred)
    # for t in np.arange(0.0, 1.0, 0.1):
    threshold = 0.5
    y_pred_tmp[y_pred > threshold] = 1.0
    y_pred_tmp[y_pred < threshold] = 0.0

    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred_tmp)

    tpr_recall = tp/(tp+fn)
    precision = tp/(tp+fp)

    f1 = 2 * tp / (2 * tp + fp + fn)
    if np.isnan(f1):
        f1 = 0.

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # return auc, f1, mcc, iou
    return tpr_recall, precision, auc, EER, f1, accuracy
