import pandas as pd
from sklearn import metrics
import numpy as np
import albumentations as albu
import cv2
import random
from scipy import stats
from sklearn.metrics import confusion_matrix
## argument library
import argparse

random.seed(24)

class puppetArgs():
    def __init__(self, dictionary: dict):
        self.d = dictionary
    def __getattr__(self, key):
        return self.d[key]

def image_transform(size = 224):
    return  albu.Compose([
                albu.PadIfNeeded(min_height = 256, min_width = 256, border_mode = cv2.BORDER_CONSTANT, value = (0, 0, 0), p = 1),
                albu.HorizontalFlip(p = 0.3),
                albu.RandomRotate90(p = 0.6),
                albu.CoarseDropout(max_holes = 60, max_height = 32, max_width = 32, min_holes = 10, min_height = 8, min_width = 8, fill_value = 0, p = 0.5),
                albu.Sequential([
                    albu.ColorJitter(brightness = 0.1, saturation = 0.3, contrast = 0.3, hue = 0, p = 0.4)
                ], p = 1),
                albu.CenterCrop(height = size, width = size, p = 1)
        ], p = 1)

def light_image_transform(size = 224):
    return  albu.Compose([
                albu.PadIfNeeded(min_height = 256, min_width = 256, border_mode = cv2.BORDER_CONSTANT, value = (0, 0, 0), p = 1),
                albu.HorizontalFlip(p = 0.3),
                albu.RandomRotate90(p = 0.6),
                albu.CoarseDropout(max_holes = 60, max_height = 32, max_width = 32, min_holes = 10, min_height = 8, min_width = 8, fill_value = 0, p = 0.5),
                albu.CenterCrop(height = size, width = size, p = 1)
        ], p = 1)

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    ---------- 
    list type, with optimal cutoff value   
    """
    fpr, tpr, thresholds = metrics.roc_curve(target, predicted)

    # method 1
    # i = np.arange(len(tpr))
    # roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    # roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    # method 2
    # threshold = thresholds[np.argmin((1 - tpr) ** 2 + fpr ** 2)]

    # method 3
    threshold = thresholds[np.argmax(tpr - fpr)]

    return fpr, tpr, metrics.auc(fpr, tpr), threshold

def performance_metrics(cfs_mtx):
    all_num = (cfs_mtx[0][0]+cfs_mtx[0][1]+cfs_mtx[1][0]+cfs_mtx[1][1])
    tpr = [ (cfs_mtx[0][0]/(cfs_mtx[0][0]+cfs_mtx[0][1])), (cfs_mtx[1][1]/(cfs_mtx[1][0]+cfs_mtx[1][1]))]
    # tpr (TP/P)
    tnr = [ (cfs_mtx[1][1]/(cfs_mtx[1][0]+cfs_mtx[1][1])), (cfs_mtx[0][0]/(cfs_mtx[0][0]+cfs_mtx[0][1]))]
    # tnr (TN/N)
    fpr = [ 1-(cfs_mtx[1][1]/(cfs_mtx[1][0]+cfs_mtx[1][1])), 1-(cfs_mtx[0][0]/(cfs_mtx[0][0]+cfs_mtx[0][1]))]
    # fpr 1-tnr = (FP/N)

    pp = [(cfs_mtx[0][0]+cfs_mtx[1][0])/all_num, (cfs_mtx[0][1]+cfs_mtx[1][1])/all_num]

    return tpr, tnr, fpr, pp, ['True Positive Rate', 'True Negative Rate', 'False Positive Rate', 'Predict Positive Rate']

def FairnessMetrics(predictions, labels, sensitives):
    AUC = []
    ACC = []
    TPR = []
    TNR = []
    PPV = []
    NPV = []
    PR = []
    NR = []
    FPR = []
    FNR = []
    TOTALACC = []

    uniSens = np.unique(sensitives)
    for modeSensitive in uniSens:
        y_pred = predictions[sensitives == modeSensitive]
        y_true = labels[sensitives == modeSensitive]
        cnf_matrix = confusion_matrix(y_true, y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        
        #AUC
        if len(set(y_true))==1: AUC.append(float("NaN"))
        else : AUC.append((metrics.roc_auc_score(y_true, y_pred)))

        # Overall accuracy for each class
        ACC.append(((TP+TN)/(TP+FP+FN+TN)).tolist()[0])
        # Sensitivity, hit rate, recall, or true positive rate
        TPR.append((TP/(TP+FN)).tolist()[0])
        # Specificity or true negative rate
        TNR.append((TN/(TN+FP)).tolist()[0])
        # Precision or positive predictive value
        PPV.append((TP/(TP+FP)).tolist()[0])
        # Negative predictive value
        NPV.append((TN/(TN+FN)).tolist()[0])
        # Fall out or false positive rate
        FPR.append((FP/(FP+TN)).tolist()[0])
        # False negative rate
        FNR.append((FN/(TP+FN)).tolist()[0])
        # Prevalence
        PR.append(((TP+FP)/(TP+FP+FN+TN)).tolist()[0])
        # Negative Prevalence
        NR.append(((TN+FN)/(TP+FP+FN+TN)).tolist()[0])
        # # False discovery rate
        # FDR = FP/(TP+FP)
        # total ACC
        TOTALACC.append(np.diag(cnf_matrix).sum()/np.sum(cnf_matrix))

    OverAll_cnf_matrix = confusion_matrix(predictions, labels)
    OverAllACC = np.diag(OverAll_cnf_matrix).sum()/np.sum(OverAll_cnf_matrix)

    AUC = np.array(AUC)
    ACC = np.array(ACC)
    TPR = np.array(TPR)
    TNR = np.array(TNR)
    PPV = np.array(PPV)
    NPV = np.array(NPV)
    PR = np.array(PR)
    NR = np.array(NR)
    FPR = np.array(FPR)
    FNR = np.array(FNR)
    TOTALACC = np.array(TOTALACC)

    return {
        'AUC': AUC.tolist(),
        'ACC': ACC.tolist(),
        'TPR': TPR.tolist(),
        'TNR': TNR.tolist(),
        'PPV': PPV.tolist(),
        'NPV': NPV.tolist(),
        'PR': PR.tolist(),
        'NR': NR.tolist(),
        'FPR': FPR.tolist(),
        'FNR': FNR.tolist(),
        'EOpp0': (TNR.max(axis = 0)-TNR.min(axis = 0)).sum(),
        'EOpp1': (TPR.max(axis = 0)-TPR.min(axis = 0)).sum(),
        'EOdd': ((TPR+FPR).max(axis = 0)-(TPR+FPR).min(axis = 0)).sum(),
        'PQD': TOTALACC.min()/TOTALACC.max(),
        'PQD(class)': (ACC.min(axis = 0)/ACC.max(axis = 0)).mean(),
        'EPPV': (PPV.min(axis = 0)/PPV.max(axis = 0)).mean(),
        'ENPV': (NPV.min(axis = 0)/NPV.max(axis = 0)).mean(),
        'DPM(Positive)': (PR.min(axis = 0)/PR.max(axis = 0)).mean(),
        'DPM(Negative)': (NR.min(axis = 0)/NR.max(axis = 0)).mean(),
        'EOM(Positive)': (TPR.min(axis = 0)/TPR.max(axis = 0)).mean(),
        'EOM(Negative)': (TNR.min(axis = 0)/TNR.max(axis = 0)).mean(),
        'OverAllAcc': OverAllACC,
        'TOTALACC': TOTALACC.tolist(),
        'TOTALACCDIF': TOTALACC.max()-TOTALACC.min(),
        'ACCDIF': (ACC.max(axis = 0)-ACC.min(axis = 0)).mean()
    }


# +
def FairnessMetricsMultiClass(predictions, labels, sensitives):
    AUC = []
    ACC = []
    TPR = []
    TNR = []
    PPV = []
    NPV = []
    PR = []
    NR = []
    FPR = []
    FNR = []
    TOTALACC = []

    uniSens = np.unique(sensitives)
    for modeSensitive in uniSens:
        y_pred = predictions[sensitives == modeSensitive]
        y_true = labels[sensitives == modeSensitive]
        cnf_matrix = confusion_matrix(y_true, y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        
        #AUC
        
#         y_true = np.argmax(y_true, axis=0)
        
#         if len(set(y_true))==1: AUC.append(float("NaN"))
#         else : AUC.append((metrics.roc_auc_score(y_true, y_pred, multi_class='ovr'))) #有問題

        AUC.append(float("NaN"))
    
        
        # Overall accuracy for each class
        ACC.append(((TP+TN)/(TP+FP+FN+TN)).tolist())
        # Sensitivity, hit rate, recall, or true positive rate
        TPR.append((TP/(TP+FN)).tolist())
        # Specificity or true negative rate
        TNR.append((TN/(TN+FP)).tolist())
        # Precision or positive predictive value
        PPV.append((TP/(TP+FP)).tolist())
        # Negative predictive value
        NPV.append((TN/(TN+FN)).tolist())
        # Fall out or false positive rate
        FPR.append((FP/(FP+TN)).tolist())
        # False negative rate
        FNR.append((FN/(TP+FN)).tolist())
        # Prevalence
        PR.append(((TP+FP)/(TP+FP+FN+TN)).tolist()[0])
        # Negative Prevalence
        NR.append(((TN+FN)/(TP+FP+FN+TN)).tolist()[0])
        # # False discovery rate
        # FDR = FP/(TP+FP)
        # total ACC
        TOTALACC.append(np.diag(cnf_matrix).sum()/np.sum(cnf_matrix))

    OverAll_cnf_matrix = confusion_matrix(labels, predictions)
    OverAllACC = np.diag(OverAll_cnf_matrix).sum()/np.sum(OverAll_cnf_matrix)

    AUC = np.array(AUC)
    ACC = np.array(ACC)
    TPR = np.array(TPR)
    TNR = np.array(TNR)
    PPV = np.array(PPV)
    NPV = np.array(NPV)
    PR = np.array(PR)
    NR = np.array(NR)
    FPR = np.array(FPR)
    FNR = np.array(FNR)
    TOTALACC = np.array(TOTALACC)

    return {
        'AUC': AUC.tolist(),
        'ACC': ACC.tolist(),
        'TPR': TPR.tolist(),
        'TNR': TNR.tolist(),
        'PPV': PPV.tolist(),
        'NPV': NPV.tolist(),
        'PR': PR.tolist(),
        'NR': NR.tolist(),
        'FPR': FPR.tolist(),
        'FNR': FNR.tolist(),
        'EOpp0': (TNR.max(axis = 0)-TNR.min(axis = 0)).sum(),
        'EOpp1': (TPR.max(axis = 0)-TPR.min(axis = 0)).sum(),
        'EOdd': ((TPR+FPR).max(axis = 0)-(TPR+FPR).min(axis = 0)).sum(),
        'PQD': TOTALACC.min()/TOTALACC.max(),
        'PQD(class)': (ACC.min(axis = 0)/ACC.max(axis = 0)).mean(),
        'EPPV': (PPV.min(axis = 0)/PPV.max(axis = 0)).mean(),
        'ENPV': (NPV.min(axis = 0)/NPV.max(axis = 0)).mean(), 
        'DPM': (PR.min(axis = 0)/PR.max(axis = 0)).mean(),
        'EOM': (TPR.min(axis = 0)/TPR.max(axis = 0)).mean(),
        'OverAllAcc': OverAllACC,
        'TOTALACC': TOTALACC.tolist(),
        'TOTALACCDIF': TOTALACC.max()-TOTALACC.min(),
        'ACCDIF': (ACC.max(axis = 0)-ACC.min(axis = 0)).mean()
    }
