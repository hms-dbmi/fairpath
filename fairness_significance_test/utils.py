import pandas as pd
from sklearn import metrics
import numpy as np
# import albumentations as albu
import cv2
import random
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
# argument library
import argparse

random.seed(24)
#################### Global Variables ####################
# list of metrics that are higher the better
HIGHER_BETTER_COLS=['AUC', 'ACC', 'TPR', 'TNR', 'PPV', 'NPV', 'PQD', 'PQD(class)','PR','NR','BAcc',
                    'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                    'EOM(Negative)', 'AUCRatio', 'OverAllAcc', 'OverAllAUC', 'TOTALACC']
# list of metrics that are lower the better
LOWER_BETTER_COLS=['FPR', 'FNR', 'EOpp0', 'EOpp1','EBAcc',
                    'EOdd', 'AUCDiff', 'TOTALACCDIF', 'ACCDIF']
# list of performance metrics
PERF_COLS=['AUC', 'ACC', 'TPR', 'TNR', 'PPV', 'NPV','PR','NR','BAcc',
            'FPR', 'FNR', 'OverAllAcc', 'OverAllAUC', 'TOTALACC']
# list of fairness metrics
FAIRNESS_COLS=['PQD', 'PQD(class)', 'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                'EOM(Negative)', 'AUCRatio', 'EOpp0', 'EOpp1','EBAcc', 'EOdd', 'AUCDiff', 'TOTALACCDIF', 'ACCDIF']
#################### Global Variables ####################


def FairnessMetrics(predictions, probs, labels, sensitives,
                    previleged_group=None, unprevileged_group=None, add_perf_difference=False):
    '''
    Estimating fairness metrics
    Args:
    * predictions: numpy array, model predictions
    * probs: numpy array, model probabilities
    * labels: numpy array, ground truth labels
    * sensitives: numpy array, sensitive attributes
    * previleged_group: str, previleged group name. If None, the group with the best performance will be used.
    * unprevileged_group: str, unprevileged group name. If None, the group with the worst performance will be used.
    * add_perf_difference: bool, whether to add performance difference metrics

    Returns:
    * results: dict, performance metrics and fairness metrics
    '''
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
    N = []
    N_0 = []
    N_1 = []
    labels = labels.astype(np.int64)
    sensitives = [str(x) for x in sensitives]
    df = pd.DataFrame({'pred': predictions, 'prob': probs,
                      'label': labels, 'group': sensitives})

    uniSens = np.unique(sensitives)
    ## categorize the groups into majority and minority groups
    if previleged_group is not None:
        if unprevileged_group is None:
            # if only the previleged group is provided, we categorize the rest of the groups as unprevileged
            group_types = ['majority' if group == previleged_group else 'minority' for group in uniSens]
        else:
            # if both previleged and unprevileged groups are provided, we categorize the groups accordingly
            group_types = ['majority' if group == previleged_group else 'minority' if group == unprevileged_group else 'unspecified' for group in uniSens]
    elif unprevileged_group is not None:
        # if only the unprevileged group is provided, we categorize the rest of the groups as previleged
        group_types = ['minority' if group == unprevileged_group else 'majority' for group in uniSens]
    else:
        # if both previleged and unprevileged groups are not provided, set to unspecified
        group_types = ['unspecified' for group in uniSens]
        
    ## calculate the performance metrics for each group
    for modeSensitive in uniSens:
        modeSensitive = str(modeSensitive)
        df_sub = df.loc[df['group'] == modeSensitive]
        y_pred = df_sub['pred'].to_numpy()
        y_prob = df_sub['prob'].to_numpy()
        y_true = df_sub['label'].to_numpy()

        if len(y_pred) == 0:
            continue
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        CR = classification_report(y_true, y_pred, labels=[
                                   0, 1], output_dict=True, zero_division=0)
        # AUC
        if len(set(y_true)) == 1:
            AUC.append(np.nan)
        else:
            AUC.append((metrics.roc_auc_score(y_true, y_prob)))
        N.append(CR['macro avg']['support'])
        N_0.append(CR['0']['support'])
        N_1.append(CR['1']['support'])
        # Overall accuracy for each class
        ACC.append(np.trace(cnf_matrix)/np.sum(cnf_matrix))
        # Sensitivity, hit rate, recall, or true positive rate
        TPR.append(CR['1']['recall'] if CR['1']['support'] > 0 else np.nan)
        # Specificity or true negative rate
        TNR.append(CR['0']['recall'] if CR['0']['support'] > 0 else np.nan)
        # Precision or positive predictive value
        PPV.append(CR['1']['precision'] if np.sum(
            cnf_matrix[:, 1]) > 0 else np.nan)
        # Negative predictive value
        NPV.append(CR['0']['precision'] if np.sum(
            cnf_matrix[:, 0]) > 0 else np.nan)
        # Fall out or false positive rate
        FPR.append(1-CR['0']['recall'] if CR['0']['support'] > 0 else np.nan)
        # False negative rate
        FNR.append(1-CR['1']['recall'] if CR['1']['support'] > 0 else np.nan)
        # Prevalence
        PR.append(np.sum(cnf_matrix[:, 1]) / np.sum(cnf_matrix))
        # Negative Prevalence
        NR.append(np.sum(cnf_matrix[:, 0]) / np.sum(cnf_matrix))
        # total ACC
        TOTALACC.append(np.trace(cnf_matrix)/np.sum(cnf_matrix))

    OverAll_cnf_matrix = confusion_matrix(predictions, labels)
    OverAllACC = np.trace(OverAll_cnf_matrix)/np.sum(OverAll_cnf_matrix)
    try:
        OverAllAUC = metrics.roc_auc_score(labels, probs)
    except:
        OverAllAUC = np.nan

    df_perf = pd.DataFrame(
        {'AUC': AUC, 'ACC': ACC, 'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 'NPV': NPV, 'BAcc': (np.array(TPR)+np.array(TNR))/2,
         'PR': PR, 'NR': NR, 'FPR': FPR, 'FNR': FNR, 'TOTALACC': TOTALACC,'OverAllAcc': OverAllACC,
         'Odd1': TPR+FPR,'Odd0':TNR+FNR,
         'OverAllAUC': OverAllAUC}, index=uniSens)
    lower_better_metrics = ['FPR', 'FNR']
    higher_better_metrics = ['TPR', 'TNR', 'NPV','BAcc',
                             'PPV', 'TOTALACC','OverAllAcc','OverAllAUC', 'AUC', 'ACC', 'PR', 'NR','Odd1','Odd0']
    
    if previleged_group is not None:
        perf_previleged = df_perf.loc[previleged_group]
    else:
        perf_previleged = pd.concat([
            df_perf[higher_better_metrics].max(),
            df_perf[lower_better_metrics].min()])
    if unprevileged_group is not None:
        perf_unprevileged = df_perf.loc[unprevileged_group]
    elif previleged_group is not None:
        perf_not_previleged = df_perf.drop(
            previleged_group)
        perf_unprevileged = pd.concat([
            perf_not_previleged[higher_better_metrics].min(),
            perf_not_previleged[lower_better_metrics].max()])
    else:
        perf_unprevileged = pd.concat([
            df_perf[higher_better_metrics].min(),
            df_perf[lower_better_metrics].max()])

    perf_diff = perf_previleged - perf_unprevileged
    perf_ratio = perf_unprevileged / perf_previleged

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
    BAcc = (TPR+TNR)/2
    
    results = {
        'sensitiveAttr': uniSens,
        'group_type': group_types,
        'N_0': N_0,
        'N_1': N_1,
        'AUC': AUC,
        'ACC': ACC,
        'TPR': TPR,
        'TNR': TNR,
        'PPV': PPV,
        'NPV': NPV,
        'BAcc': BAcc,
        'PR': PR,
        'NR': NR,
        'FPR': FPR,
        'FNR': FNR,
        'EOpp0': perf_diff['TNR'],
        'EOpp1': perf_diff['TPR'],
        'EBAcc': perf_diff['BAcc'],
        'EOdd':  (-perf_diff['Odd1']),
        'EOdd0':  (-perf_diff['Odd0']),
        'EOdd1':  (-perf_diff['Odd1']),
        'PQD': perf_ratio['TOTALACC'],
        'PQD(class)': perf_ratio['TOTALACC'],
        'EPPV': perf_ratio['PPV'],
        'ENPV': perf_ratio['NPV'],
        'DPM(Positive)': perf_ratio['PR'],
        'DPM(Negative)': perf_ratio['NR'],
        'EOM(Positive)': perf_ratio['TPR'],
        'EOM(Negative)':  perf_ratio['TNR'],
        'AUCRatio':  perf_ratio['AUC'],
        'AUCDiff':  perf_diff['AUC'],
        'OverAllAcc': OverAllACC,
        'OverAllAUC': OverAllAUC,
        'TOTALACC': TOTALACC,
        'TOTALACCDIF': perf_diff['TOTALACC'],
        'ACCDIF': perf_diff['ACC'],
    }
    if add_perf_difference:
        results, new_cols = get_perf_diff(
            results,perf_metrics=PERF_COLS,
            privileged_group=previleged_group,demo_col='sensitiveAttr')
    return results

def get_metric_names(add_perf_difference=False):
    '''
    returns a dictionary of metric list
    '''
    metrics_list = {
        'higher_better_metrics': HIGHER_BETTER_COLS.copy(),
        'lower_better_metrics': LOWER_BETTER_COLS.copy(),
        'perf_metrics': PERF_COLS.copy(),
        'fairness_metrics': FAIRNESS_COLS.copy()
    }
    ## if add_perf_difference == True, we add the performance difference as a fairness metric
    if add_perf_difference:
        for col in PERF_COLS:
            metrics_list['fairness_metrics'].append(f'{col}_diff')
            if col in HIGHER_BETTER_COLS:
                metrics_list['lower_better_metrics'].append(f'{col}_diff')
            elif col in LOWER_BETTER_COLS:
                metrics_list['higher_better_metrics'].append(f'{col}_diff')
    metrics_list['higher_better_fairness_metrics'] = [x for x in metrics_list['fairness_metrics'] if x in metrics_list['higher_better_metrics']]
    metrics_list['lower_better_fairness_metrics'] = [x for x in metrics_list['fairness_metrics'] if x in  metrics_list['lower_better_metrics']]
    return metrics_list


def get_perf_diff(fairResult,perf_metrics=PERF_COLS,privileged_group=None,demo_col = 'sensitiveAttr'):
    '''
    Add performance difference to the dataframe
    '''
    df = pd.DataFrame(fairResult)
    new_cols = []
    for col in perf_metrics:
        if col in df.keys():
            if privileged_group is not None:
                val_privileged = df.loc[df[demo_col]==privileged_group][col]
                val_others = df.loc[df[demo_col]!=privileged_group][col]
            else:
                val_privileged = df[col]
                val_others = df[col]
            val_privileged = val_privileged.max()
            val_others = val_others.min()
            
            
            # df[f'{col}Diff'] = val_privileged - val_others
            fairResult[col+'_diff'] = val_privileged - val_others
            new_cols.append(col+'_diff')
            
    return fairResult, new_cols
