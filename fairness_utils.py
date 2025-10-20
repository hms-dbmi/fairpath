import pandas as pd
from sklearn import metrics
import numpy as np
# import albumentations as albu
import random
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
# argument library
import argparse

random.seed(24)

PERF_COLS_NO_OVERALL=['AUC', 'ACC', 'TPR', 'TNR', 'PPV', 'NPV','PR','NR','BAcc',
            'TOTALACC', 'F1_0', 'F1_1', 'AUPRC_0', 'AUPRC_1', 'KAPPA', 'MCC']
PERF_COLS=PERF_COLS_NO_OVERALL + [f'OverAll{x}' for x in PERF_COLS_NO_OVERALL]

HIGHER_BETTER_COLS=PERF_COLS + ['PQD', 'PQD(class)',
                    'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                    'EOM(Negative)', 'AUCRatio']
LOWER_BETTER_COLS=['EOpp0', 'EOpp1','EBAcc',
                    'EOdd', 'EOddAbs','EOddMax','AUCDiff', 'TOTALACCDIF', 'ACCDIF']

FAIRNESS_COLS=['PQD', 'PQD(class)', 'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                'EOM(Negative)', 'AUCRatio', 'EOpp0', 'EOpp1','EBAcc', 'EOdd', 'EOddAbs','EOddMax','AUCDiff', 'TOTALACCDIF', 'ACCDIF']
## metric pair that evaluates individual classes
SINGLE_CLASS_METRICS_DICTS = {
    'EOpp': ['EOpp0', 'EOpp1'],
    'EPrecision': ['EPPV', 'ENPV'],
    'DPM': ['DPM(Positive)', 'DPM(Negative)'],
    'EOM': ['EOM(Positive)', 'EOM(Negative)'],
    'Recall_diff': ['TPR_diff', 'TNR_diff'],
    'Precision_diff': ['PPV_diff', 'NPV_diff'],
    'F1_diff': ['F1_0_diff', 'F1_1_diff'],
    'AUPRC_diff': ['AUPRC_0_diff', 'AUPRC_1_diff'],
}

# maps the csv name to the TCGA project name
TCGA_NAME_DICT = {
    # tumor detection
    'LUAD_TumorDetection':  '04_LUAD',
    'CCRCC_TumorDetection':  '06_KIRC',
    'HNSC_TumorDetection':  '07_HNSC',
    'LSCC_TumorDetection':  '10_LUSC',
    # 'BRCA_TumorDetection':  '01_BRCA',
    'PDA_TumorDetection':  '11_PRAD',
    'UCEC_TumorDetection':  '05_UCEC',
    # cancer type classification
    'COAD_READ_512': '_COAD+READ',
    'KIRC_KICH_512': '_KIRC+KICH',
    'KIRP_KICH_512': '_KIRP+KICH',
    'KIRC_KIRP_512': '_KIRC+KIRP',   
    'LGG_GBM_512': '_GBM+LGG',
    'LUAD_LUSC_512': '_LUAD+LUSC',
    'COAD_READ': '_COAD+READ',
    'KIRC_KICH': '_KIRC+KICH',
    'KIRP_KICH': '_KIRP+KICH',
    'KIRC_KIRP': '_KIRC+KIRP',   
    'LGG_GBM': '_GBM+LGG',
    'LUAD_LUSC': '_LUAD+LUSC',
    # cancer subtype classification
    'Breast_ductal_lobular_512': '01_BRCA 1+1',
    'LUAD_BRONCHIOLO-ALVEOLAR_512': '04_LUAD 3+n',
    'Breast_ductal_lobular': '01_BRCA 1+1',
    'LUAD_BRONCHIOLO-ALVEOLAR': '04_LUAD 3+n',
    'LUAD_3_n': '04_LUAD 3+n',
}

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


def performance_metrics(df):

    y_pred = df['pred'].to_numpy()
    y_prob = df['prob'].to_numpy()
    y_true = df['label'].to_numpy()

    if len(y_pred) == 0:
        return None
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    CR = classification_report(y_true, y_pred, labels=[
                                0, 1], output_dict=True, zero_division=0)
    perf_results = {
        'N': CR['macro avg']['support'],
        'N_0': CR['0']['support'],
        'N_1': CR['1']['support'],
        'AUC': metrics.roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        'ACC': np.trace(cnf_matrix)/np.sum(cnf_matrix),
        'TPR': CR['1']['recall'] if CR['1']['support'] > 0 else np.nan,
        'TNR': CR['0']['recall'] if CR['0']['support'] > 0 else np.nan,
        'BAcc': (CR['1']['recall'] + CR['0']['recall'])/2 if CR['1']['support'] > 0 and CR['0']['support'] > 0 else np.nan,
        'PPV': CR['1']['precision'] if np.sum(cnf_matrix[:, 1]) > 0 else np.nan,
        'NPV': CR['0']['precision'] if np.sum(cnf_matrix[:, 0]) > 0 else np.nan,
        # 'FPR': 1-CR['0']['recall'] if CR['0']['support'] > 0 else np.nan,
        # 'FNR': 1-CR['1']['recall'] if CR['1']['support'] > 0 else np.nan,
        'PR': np.sum(cnf_matrix[:, 1]) / np.sum(cnf_matrix),
        'NR': np.sum(cnf_matrix[:, 0]) / np.sum(cnf_matrix),
        'Odd1': CR['1']['recall'] + 1-CR['0']['recall'],
        'Odd0': CR['0']['recall'] + 1-CR['1']['recall'],
        'TOTALACC': np.trace(cnf_matrix)/np.sum(cnf_matrix),
        'F1_0': CR['0']['f1-score'],
        'F1_1': CR['1']['f1-score'],
        'AUPRC_0': metrics.average_precision_score(y_true, y_prob, pos_label=0),
        'AUPRC_1': metrics.average_precision_score(y_true, y_prob, pos_label=1),
        'KAPPA': metrics.cohen_kappa_score(y_true, y_pred),
        'MCC': metrics.matthews_corrcoef(y_true, y_pred)
    }
    return perf_results



def FairnessMetrics(predictions, probs, labels, sensitives,
                    previleged_group=None, unprevileged_group=None,
                    add_combined_metrics=True,
                    add_perf_difference=False):
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
    ##
    group_perf_dicts = []
    ## performance metrics for each group
    for modeSensitive in uniSens:
        modeSensitive = str(modeSensitive)
        df_sub = df.loc[df['group'] == modeSensitive]
        if len(df_sub) == 0:
            continue
        group_perf_dicts.append(performance_metrics(df_sub))
    keys = group_perf_dicts[0].keys()
    # combine the performance metrics for each group
    group_perf_dict = {key: [d[key] for d in group_perf_dicts] for key in keys}
    ## performance metrics for the overall dataset
    overall_perf_dict = performance_metrics(df)
    ##
    # perf_dict = {**group_perf_dict, **overall_perf_dict}
    ##
    df_perf = pd.DataFrame(group_perf_dict, index=uniSens)
    # df_overall_perf = pd.DataFrame(overall_perf_dict, index=['Overall'])

    # higher_better_metrics = ['TPR', 'TNR', 'NPV','BAcc',
    #                         'PPV', 'TOTALACC', 'AUC', 'ACC', 'PR', 'NR','Odd1','Odd0',
    #                         'F1_0', 'F1_1', 'AUPRC_0', 'AUPRC_1', 'KAPPA', 'MCC']
    higher_better_metrics = PERF_COLS_NO_OVERALL + ['Odd1','Odd0']
    # lower_better_metrics = lower_better_metrics + ['OverAll'+x for x in lower_better_metrics]
    # higher_better_metrics = higher_better_metrics + ['OverAll'+x for x in higher_better_metrics]
    
    if previleged_group is not None:
        perf_previleged = df_perf.loc[previleged_group]
    else:
        # if previleged_group is not provided, we take the group with the best performance
        perf_previleged = df_perf[higher_better_metrics].max()
        
    if unprevileged_group is not None:
        perf_unprevileged = df_perf.loc[unprevileged_group]
    elif previleged_group is not None:
        perf_not_previleged = df_perf.drop(
            previleged_group)
        perf_unprevileged = perf_not_previleged[higher_better_metrics].min()
    else:
        # if unprevileged_group is not provided, we take the group with the worst performance
        perf_unprevileged = df_perf[higher_better_metrics].min()

    perf_diff = perf_previleged - perf_unprevileged
    perf_ratio = perf_unprevileged / perf_previleged

    fairness_dict = {
        'EOpp0': perf_diff['TNR'],
        'EOpp1': perf_diff['TPR'],
        'EBAcc': perf_diff['BAcc'],
        'EOdd':  (-perf_diff['Odd1']), # equivalent to aif360.metrics.average_odds_difference()
        'EOddAbs': (np.abs(perf_diff['TNR']) + np.abs(perf_diff['TPR']))/2, # equivalent to aif360.metrics.average_abs_odds_difference()
        'EOddMax':  np.max([np.abs(perf_diff['TNR']), np.abs(perf_diff['TPR'])]),   # equivalent to aif360.metrics.equalized_odds_difference()
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
        'TOTALACCDIF': perf_diff['TOTALACC'],
        'ACCDIF': perf_diff['ACC'],
    }
    remove_keys = ['Odd1','Odd0']
    for key in remove_keys:
        del perf_diff[key]
        del perf_ratio[key]
        del group_perf_dict[key]
        del overall_perf_dict[key]
        
    overall_perf_dict = {f'OverAll{key}': val for key, val in overall_perf_dict.items()}
    results = {
        'sensitiveAttr': uniSens,
        'group_type': group_types,
        **group_perf_dict,
        **overall_perf_dict,
        **fairness_dict}
    if add_perf_difference:
        perf_diff_dict = perf_diff.to_dict()
        perf_diff_dict = {f'{key}_diff': perf_diff_dict[key] for key in perf_diff_dict}
        results.update(perf_diff_dict)
    if add_combined_metrics:
        for key, value in SINGLE_CLASS_METRICS_DICTS.items():
            if all(x in results for x in value):
                results[f'{key}_avg'] = np.mean([results[x] for x in value])
                if value[0] in HIGHER_BETTER_COLS:
                    results[f'{key}_min'] = np.min([results[x] for x in value])
                else:
                    results[f'{key}_max'] = np.max([results[x] for x in value])

    return results

def get_metric_names(add_perf_difference=False,add_combined_metrics=True):
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
        for col in PERF_COLS_NO_OVERALL:
            metrics_list['fairness_metrics'].append(f'{col}_diff')
            if col in HIGHER_BETTER_COLS:
                metrics_list['lower_better_metrics'].append(f'{col}_diff')
            elif col in LOWER_BETTER_COLS:
                metrics_list['higher_better_metrics'].append(f'{col}_diff')
    
    if add_combined_metrics:
        for key, value in SINGLE_CLASS_METRICS_DICTS.items():
            if value[0] in metrics_list['higher_better_metrics']:
                metrics_list['higher_better_metrics'].extend([f'{key}_avg', f'{key}_min'])     
                metrics_list['fairness_metrics'].extend([f'{key}_avg', f'{key}_min'])           
            elif value[0] in metrics_list['lower_better_metrics']:
                metrics_list['lower_better_metrics'].extend([f'{key}_avg', f'{key}_max'])
                metrics_list['fairness_metrics'].extend([f'{key}_avg', f'{key}_max'])
    metrics_list['higher_better_fairness_metrics'] = [x for x in metrics_list['fairness_metrics'] if x in metrics_list['higher_better_metrics']]
    metrics_list['lower_better_fairness_metrics'] = [x for x in metrics_list['fairness_metrics'] if x in  metrics_list['lower_better_metrics']]
    return metrics_list

# for inpath in BASES:
def get_perf_diff(fairResult,perf_metrics=PERF_COLS,privileged_group=None,demo_col = 'sensitiveAttr'):
    '''
    Add performance metric to the dataframe
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


def FairnessMetricsMultiClass(predictions, labels, sensitives):
    # TODO: fix the bug like FairnessMetrics()
    raise NotImplementedError
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
        'ACC': ACC.tolist(),
        'TPR': TPR.tolist(),
        'TNR': TNR.tolist(),
        'PPV': PPV.tolist(),
        'NPV': NPV.tolist(),
        'PR': PR.tolist(),
        'NR': NR.tolist(),
        'FPR': FPR.tolist(),
        'FNR': FNR.tolist(),
        'EOpp0': (TNR.max(axis=0)-TNR.min(axis=0)).sum(),
        'EOpp1': (TPR.max(axis=0)-TPR.min(axis=0)).sum(),
        'EOdd': ((TPR+FPR).max(axis=0)-(TPR+FPR).min(axis=0)).sum(),
        'PQD': TOTALACC.min()/TOTALACC.max(),
        'PQD(class)': (ACC.min(axis=0)/ACC.max(axis=0)).mean(),
        'EPPV': (PPV.min(axis=0)/PPV.max(axis=0)).mean(),
        'ENPV': (NPV.min(axis=0)/NPV.max(axis=0)).mean(),
        'DPM': (PR.min(axis=0)/PR.max(axis=0)).mean(),
        'EOM': (TPR.min(axis=0)/TPR.max(axis=0)).mean(),
        'OverAllAcc': OverAllACC,
        'TOTALACC': TOTALACC.tolist(),
        'TOTALACCDIF': TOTALACC.max()-TOTALACC.min(),
        'ACCDIF': (ACC.max(axis=0)-ACC.min(axis=0)).mean()
    }
