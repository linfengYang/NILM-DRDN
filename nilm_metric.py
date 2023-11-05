#from Logger import log
import numpy as np
import pandas as pd
# from sklearn.metrics import confusion_matrix
from Arguments import params_appliance_UKDALE


def get_EA(actuals,predicts):
    return 1-abs(actuals-predicts).sum()/(2*actuals.sum())


def get_TP(target, prediction, threshold):
    '''
    compute the  number of true positive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = np.where(target > threshold, 1, 0)
    prediction = np.where(prediction > threshold, 1, 0)

    tp_array = np.logical_and(target, prediction) * 1.0
    tp = np.sum(tp_array)

    return tp


def get_FP(target, prediction, threshold):
    '''
    compute the  number of false positive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = 1 - np.where(target > threshold, 1, 0)
    prediction = np.where(prediction > threshold, 1, 0)

    fp_array = np.logical_and(target, prediction) * 1.0
    fp = np.sum(fp_array)

    return fp


def get_FN(target, prediction, threshold):
    '''
    compute the  number of false negtive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = np.where(target > threshold, 1, 0)
    prediction = 1 - np.where(prediction > threshold, 1, 0)

    fn_array = np.logical_and(target, prediction) * 1.0
    fn = np.sum(fn_array)

    return fn


def get_TN(target, prediction, threshold):
    '''
    compute the  number of true negative

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = 1 - np.where(target > threshold, 1, 0)
    prediction = 1 - np.where(prediction > threshold, 1, 0)

    tn_array = np.logical_and(target, prediction) * 1.0
    tn = np.sum(tn_array)

    return tn


def get_recall(target, prediction, threshold):
    '''
    compute the recall rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    fn = get_FN(target, prediction, threshold)
    if tp + fn <= 0.0:
        recall = tp / (tp + fn + 1e-9)
    else:
        recall = tp / (tp + fn)
    return recall


def get_precision(target, prediction, threshold):
    '''
    compute the  precision rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    fp = get_FP(target, prediction, threshold)
    if tp + fp <= 0.0:
        precision = tp / (tp + fp + 1e-9)
    else:
        precision = tp / (tp + fp)
    return precision


def get_F1(target, prediction, threshold):
    '''
    compute the  F1 score

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    recall = get_recall(target, prediction, threshold)
    precision = get_precision(target, prediction, threshold)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_accuracy(target, prediction, threshold):
    '''
    compute the accuracy rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    tn = get_TN(target, prediction, threshold)

    accuracy = (tp + tn) / target.size

    return accuracy


def get_relative_error(target, prediction):
    '''
    compute the  relative_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    assert (target.shape == prediction.shape)

    return np.mean(np.nan_to_num(np.abs(target - prediction) / np.maximum(target, prediction)))


def get_abs_error(target, prediction):
    '''
    compute the  absolute_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    assert (target.shape == prediction.shape)

    data = np.abs(target - prediction)
    mean, std, min_v, max_v, quartile1, median, quartile2 = get_statistics(data)

    return mean, std, min_v, max_v, quartile1, median, quartile2, data


def get_nde(target, prediction):
    '''
    compute the  normalized disaggregation error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    return np.sum((target - prediction) ** 2) / np.sum((target ** 2))


def get_sae(target, prediction, sample_second):
    '''
    compute the signal aggregate error
    sae = |\hat(r)-r|/r where r is the ground truth total energy;
    \hat(r) is the predicted total energy.
    '''
    r = np.sum(target * sample_second * 1.0 / 3600.0)
    rhat = np.sum(prediction * sample_second * 1.0 / 3600.0)

    sae = np.abs(r - rhat) / np.abs(r)

    return sae

def get_Epd(target, prediction, sample_second):
    '''
    Energy per day
    - calculate energy of a day for both ground truth and prediction
    - sum all the energies
    - divide by the number of days
    '''

    day = int(24.0 * 3600 / sample_second)
    gt_en_days = []
    pred_en_days = []

    for start in range(0, int(len(target)-day), int(day)):
        gt_en_days.append(np.sum(target[start:start+day]*sample_second)/3600)
        pred_en_days.append(np.sum(prediction[start:start+day]*sample_second)/3600)

    Epd = np.sum(np.abs(np.array(gt_en_days)-np.array(pred_en_days)))/(len(target)/day)

    return Epd


def get_statistics(data):

    mean = np.mean(data)
    std = np.std(data)
    min_v = np.sort(data)[0]
    max_v = np.sort(data)[-1]

    quartile1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    quartile2 = np.percentile(data, 75)

    return mean, std, min_v, max_v, quartile1, median, quartile2


def tp_tn_fp_fn(states_pred, states_ground):
    tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
    fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
    fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
    tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
    return tp, tn, fp, fn

def recall_precision_accuracy_f1(target, prediction,threshold):
    recall = get_recall(target, prediction, threshold)
    precision = get_precision(target, prediction, threshold)
    f1 = get_F1(target, prediction, threshold)
    accuracy = get_accuracy(target, prediction, threshold)
    return recall,precision,f1,accuracy

def relative_error_total_energy(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        E_pred = sum(chunk.iloc[:,0])
        E_ground = sum(chunk.iloc[:,1])

        chunk_results.append([
                            E_pred,
                            E_ground
                            ])
    if sum_samples == 0:
        return None
    else:
        [E_pred, E_ground] = np.sum(chunk_results,axis=0)
        return abs(E_pred - E_ground) / float(max(E_pred,E_ground))

def mean_absolute_error(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    total_sum = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        total_sum += sum(abs((chunk.iloc[:,0]) - chunk.iloc[:,1]))
    if sum_samples == 0:
        return None
    else:
        return total_sum / sum_samples


def recall(tp,fn):
    return tp/float(tp+fn)

def precision(tp,fp):
    return tp/float(tp+fp)

def f1(prec,rec):
    return 2 * (prec*rec) / float(prec+rec)

def accuracy(tp, tn, p, n):
    return (tp + tn) / float(p + n)

def get_metric(actuals,predicts,app_name):
    EA = get_EA(actuals, predicts)
    SAE = get_sae(actuals, predicts, 6)
    MAE, std, min_v, max_v, quartile1, median, quartile2, data = get_abs_error(actuals, predicts)
    recall, precision, f1, accuracy = recall_precision_accuracy_f1(target=actuals, prediction=predicts,
                                                                   threshold=params_appliance_UKDALE[app_name][
                                                                       'on_power_threshold'])
    return recall, precision, f1, accuracy, MAE, SAE, EA