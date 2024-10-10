
'''
Script to calculate the bias of models using bootstrap significance test
'''

import pandas as pd
import json
import os
from os.path import join, basename, dirname
from scipy.stats import combine_pvalues
import glob
import numpy as np
from argparse import ArgumentParser
from typing import List, Tuple, Literal
from utils import FairnessMetrics, get_metric_names

from tqdm import tqdm

SEED=0
SUBJ_ID_COL = 'bcr_patient_barcode'
EXCLUDE_RACES = ['AMERICAN INDIAN OR ALASKA NATIVE', '[not available]']
MINORITY_RACES = ['BLACK OR AFRICAN AMERICAN', 'ASIAN']


# parse arguments
parser = ArgumentParser()
parser.add_argument('--results_folder', type=str, help='Directory to the results folder for TCGA. Should be in the format of <results_folder>/<task_results_folder>',
                    default='/n/data2/hms/dbmi/kyu/lab/NCKU/Fairness/baseline_model'
                    )
parser.add_argument('--output_folder', type=str, help='directory to the output folder in google drive',default='debug')
parser.add_argument('--n_bootstrap', type=int,
                    help='number of bootstrap iterations', default=10000)
parser.add_argument('--aggregate_method', type=str, choices=['concatenate','fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george','groupwise'],
                    help='method to aggregate p-values', default='concatenate')
parser.add_argument('--add_perf_difference', action='store_true', help='add performance difference as a fairness metric', default=True)
parser.add_argument('--fix_privileged_group', action='store_true', help='Use predefined privileged groups', default=False)
parser.add_argument('--not_fix_privileged_group',help='Don not use predefined privileged groups',action='store_false',dest='fix_privileged_group')

args = parser.parse_args()
# end of parsing arguments
METRIC_NAMES_DICT = get_metric_names(add_perf_difference=args.add_perf_difference)


def get_TCGA_demo_info(
        demo_col,
        demo_folder='TCGA_all_clinical'):
    '''
    get the demographic information from the clinical files of TCGA.
    All clinical file can be downloaded from GDC data portal.
    input:
        demo_col: str, the demographic attribute
        demo_folder: str, the folder containing the clinical files from GDC data portal
    '''
    demo_csvs = glob.glob(
        join(demo_folder, 'nationwidechildrens.org_clinical_patient_*.txt'))
    dfs_demo = []
    for f in demo_csvs:
        df = pd.read_csv(f, delimiter='\t')
        df = df.iloc[2:]
        age_col = 'days_to_birth' if 'days_to_birth' in df.columns else 'birth_days_to'
        df['age'] = df[age_col]
        dfs_demo.append(df)

    df_demo = pd.concat(dfs_demo)
    df_demo = df_demo[[SUBJ_ID_COL, demo_col]]
    if demo_col == 'race':
        # exclude racial groups with insufficient sample size
        df_demo = df_demo[~df_demo[demo_col].isin(EXCLUDE_RACES)]
    df_demo['sens_attr'] = df_demo.pop(demo_col)
    return df_demo


def compile_dataframe(labels, preds, probs, df_demo,demo_col):
    '''
    compile the dataframe
    input:
        labels: pd.Series, the ground truth labels
        preds: pd.Series, the predictions
        df_demo: pd.DataFrame, the demographic information
        demo_col: str, the demographic attribute
    output:
        df: pd.DataFrame, the compiled dataframe
    '''
    df = pd.concat([labels, preds, probs], axis=1).reset_index()
    df['slide'] = df.pop('index')
    df['pred'] = [i[1] for i in df['pred']]  # get the positive class
    # get TCGA ID
    df[SUBJ_ID_COL] = ['-'.join(i.split('-')[:3]) for i in df['slide']]
    # .reset_index(drop=True)
    df = df.merge(df_demo, on=SUBJ_ID_COL, how='left')
    ## process age
    
    NA_STRS = ['[Not Available]', '[Not Applicable]',
                '[Not Evaluated]', '[Completed]']
    df.replace(NA_STRS, np.nan, inplace=True)
    df = df.dropna(subset=['sens_attr'])
    if demo_col == 'age':
        df['sens_attr'] = df['sens_attr'].astype(float)/-365.25
    
    return df


def get_results_from_folder(folder):
    '''
    get the results from the folder (for new data)
    input:
        folder: str, the folder containing the results
    output:
        df: pd.DataFrame, the results. It contains the following columns:
            - 'label': the ground truth label
            - 'pred': the prediction
            - 'sens_attr': the demographic attribute
        privileged_group: str, the privileged group
    '''
    attr = basename(folder).split(' ')[0]
    if attr == 'female':
        demo_col = 'gender'
    elif attr == 'white':
        demo_col = 'race'
    else:
        demo_col = 'age'
    # find data
    df_demo = get_TCGA_demo_info(demo_col)

    # label
    label_jsons = [
        join(folder, f'groundTruthResults_test{fold}_0.json') for fold in range(4)]
    # prediction
    # NOTE: this json file only contains binary prediction, not the probability
    # if you want to use the probability, you need to change the json file to load the probability
    pred_jsons = [
        join(folder, f'predResults_test{fold}_0.json') for fold in range(4)]
    
    prob_npys = [join(folder, f'AUCprob_test{fold}_0.npy') for fold in range(4)]
    assert all([os.path.exists(f) for f in label_jsons+pred_jsons+prob_npys]), f'Error: some files do not exist in {folder}'
    
    ######### process the data
    dfs = []
    for fold, pred_file, label_file,prob_npy in enumerate(zip(pred_jsons, label_jsons,prob_npys)):
        pred_d = json.load(open(pred_file))
        label_d = json.load(open(label_file))
        prob_M = np.load(prob_npy)
        prob_d = {key:value for key,value in zip(label_d.keys(),prob_M)}
        pred = pd.Series(pred_d)
        label = pd.Series(label_d)
        prob = pd.Series(prob_d)
        df = compile_dataframe(label, pred, prob,df_demo,demo_col)
        dfs.append(df)

        
    return dfs, demo_col


def CV_bootstrap_bias_test(
    dfs, privileged_group=None, n_bootstrap=1000,aggregate_method='fisher'):
    '''
    Estimate the improvement of the corrected model over the baseline model for all folds
    Input:
        dfs: list of pd.DataFrame, the results
        n_bootstrap: int, number of bootstrap iterations
        aggregate_method: str, method to aggregate p-values. Options are:
            - 'concatenate': concatenate the input data
            - 'fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george': methods to combine p-values, see scipy.stats.combine_pvalues for details
    Output:
        df_p_better: pd.DataFrame, the p-values for significant improvement
        df_p_worse: pd.DataFrame, the p-values for significant worsening
        
    '''
    
    if aggregate_method == 'concatenate':
        # if the method is concatenate, we concatenate the data and return a single p-value
        df = pd.concat(dfs).reset_index(drop=True)

        fairResult, df_p_worse, df_p_better = bootstrap_bias_test(df,
                        bootstrap_n=n_bootstrap, privileged_group=privileged_group)
        return df_p_worse, df_p_better, fairResult
    elif aggregate_method in ['fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george']:
        # if the method is fisher or stouffer, we calculate the p-value for each fold
        dfs_p_better = []
        dfs_p_worse = []
        fairResult_list = []
        for i, df in enumerate(dfs):
            fairResult, df_p_worse, df_p_better = bootstrap_bias_test(df,
                            bootstrap_n=n_bootstrap, privileged_group=privileged_group)
            df_p_better.insert(0, 'fold', i)
            df_p_worse.insert(0, 'fold', i)
            fairResult.insert(0, 'fold', i)
            dfs_p_better.append(df_p_better)
            dfs_p_worse.append(df_p_worse)
            fairResult_list.append(fairResult)
        ## concatenate the p-values
        df_p_better = pd.concat(dfs_p_better)
        df_p_worse = pd.concat(dfs_p_worse)
        fairResult = pd.concat(fairResult_list)
        ## aggregate the p-values
        df_p_combined = dfs_p_better[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_better[col].loc[i]
                meta_res = combine_pvalues(pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_better = pd.concat([df_p_better, df_p_combined])
        
        df_p_combined = dfs_p_worse[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_worse[col].loc[i]
                meta_res = combine_pvalues(pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_worse = pd.concat([df_p_worse, df_p_combined])
        return df_p_worse, df_p_better, fairResult

    elif aggregate_method == 'groupwise':
        # if the method is groupwise, we estimate the fairness metrics first, and then perform bootstraping on population level
        df_perf_avgdiff, df_p_worse, df_p_better = bootstrap_bias_test_groupLevel(dfs,
                            bootstrap_n=n_bootstrap, privileged_group=privileged_group)

        return df_p_worse, df_p_better, df_perf_avgdiff

        
def get_mean_perf_diff(df_perf):
    dfs_group = {group: df_perf.groupby(['group_type']).get_group(group) for group in ['majority','minority']}
    df_diff = dfs_group['majority'][METRIC_NAMES_DICT['perf_metrics']] - dfs_group['minority'][METRIC_NAMES_DICT['perf_metrics']]
    # for metrics that are lower the better, we reverse the sign
    lower_better_perf = list(set(METRIC_NAMES_DICT['lower_better_metrics']).intersection(set(df_diff.columns)))
    # df_diff[lower_better_perf] = -df_diff[lower_better_perf]
    return df_diff.mean(), df_diff
    
def bootstrap_bias_test_groupLevel(dfs,
                        bootstrap_n=1000, privileged_group=None,add_perf_difference=True):
    '''
    Do the bootstrap test on the group level (bootstrapping on gorup-level performance metrics)
    H0: mean performance difference between the unprivileged group and the privileged group (across all folds) is 0 
    Ha: > 0
    '''
    ## get the performance metrics for each fold
    fairResult_list = []
    for i, df in enumerate(dfs):
        fairResult = FairnessMetrics(
            df[f'pred'].to_numpy(),
            df[f'prob'].to_numpy(),
            df['label'].to_numpy(),
            df['sens_attr'].astype(str).to_numpy(),
            previleged_group=privileged_group,add_perf_difference=True)
        fairResult['fold'] = i
        fairResult = pd.DataFrame(fairResult)
        fairResult_list.append(fairResult)
    # df = pd.DataFrame.from_records(fairResult_list)
    df = pd.concat(fairResult_list)
    df_perf = df[['sensitiveAttr','group_type','fold'] + METRIC_NAMES_DICT['perf_metrics']]
    df_perf = df_perf.set_index(['fold'],drop=True)
    df_perf_groupwise = df_perf.pivot(columns='sensitiveAttr').reset_index(drop=True)
    df_perf_groupwise.drop('group_type',axis=1,inplace=True)
    df_perf_mean = df_perf_groupwise.mean(axis=0)
    ## get performance difference (positive is biased against the unprivileged group)
    df_perf_avgdiff,df_perf_diff = get_mean_perf_diff(df_perf)
    
    
    
    ## get the bootstrap samples
    bootstrap_results = []
    np.random.seed(SEED)

    for i in tqdm(range(bootstrap_n),miniters=bootstrap_n//10):
        # bootstrap within each fold
        df_shuffled = df_perf.groupby('fold').sample(frac=1,replace=True)#.reset_index(drop=True)
        df_shuffled[['sensitiveAttr','group_type']] = df_perf[['sensitiveAttr','group_type']]  # add the sensitive attribute
        df_shuffled_avgdiff,_ = get_mean_perf_diff(df_shuffled)
        bootstrap_results.append(df_shuffled_avgdiff)
    df_bootstrap = pd.concat(bootstrap_results,axis=1).T
    
    ########################################################
    # get the percentile of the measured values over bootstraped samples
    # p-value of the fairness metrics (biased against the unprevileged group)
    p_table_worse = {}
    # p-value of the fairness metrics (biased in favor of the unprevileged group)
    p_table_better = {}
    for metric in METRIC_NAMES_DICT['perf_metrics']:
        measure = df_perf_avgdiff[metric]
        n_valid_bootstrap = df_bootstrap[metric].notna().sum()
        if n_valid_bootstrap < bootstrap_n:
            print(
                f'Warning: {n_valid_bootstrap} valid bootstrap samples for {metric}. Expected {bootstrap_n}')
        if metric in METRIC_NAMES_DICT['higher_better_metrics']:
            p_table_worse[metric] = (df_bootstrap[metric] >=
                                     measure).sum() / n_valid_bootstrap
            p_table_better[metric] = (df_bootstrap[metric] <=
                                      measure).sum() / n_valid_bootstrap
        else:
            p_table_worse[metric] = (df_bootstrap[metric] <=
                                     measure).sum() / n_valid_bootstrap
            p_table_better[metric] = (df_bootstrap[metric] >=
                                      measure).sum() / n_valid_bootstrap
    p_table_worse = pd.Series(
        p_table_worse)
    p_table_better = pd.Series(
        p_table_better)

    p_table_worse = pd.DataFrame(p_table_worse, columns=['pval']).transpose()
    p_table_better = pd.DataFrame(p_table_better, columns=['pval']).transpose()
    # fairResult = pd.DataFrame(fairResult).transpose()
    
        # df_shuffled = df_perf.sample(
        #     frac=1, replace=True).reset_index(drop=True)
    df_perf_avgdiff = df_perf_avgdiff.to_frame().T
    df_perf_avgdiff.columns = [f'{x}_diff' for x in df_perf_avgdiff.columns]
    p_table_worse.columns = [f'{x}_diff' for x in p_table_worse.columns]
    p_table_better.columns = [f'{x}_diff' for x in p_table_better.columns]
    return fairResult, p_table_worse, p_table_better
    
            
            
def bootstrap_bias_test(df,
                        bootstrap_n=1000, privileged_group=None,add_perf_difference=True):
    '''
    Estimate the p-value of the fairness metrics using bootstrap
    Args:
    * df: pandas DataFrame, data, including the following columns:
        * sens_attr: sensitive attributes
        * prob: model probabilities
        * label: ground truth labels
        * pred:  model predictions
    * bootstrap_n: int, number of bootstrap samples
    * privileged_group: str, previleged group name. If None, the group with the best performance will be used.
    Returns:
    * fairResult: dict, performance metrics and fairness metrics
    * p_table_worse: pandas Series, p-value of the fairness metrics 
        (biased against the underprivileged group, if the privileged group is defined)
        (otherwise, biased against the group with the lowest performance)
    * p_table_better: pandas Series, p-value of the fairness metrics (biased against the privileged group)
    '''
    df = df[['sens_attr', 'prob', 'label', 'pred']]
    df_g = df['sens_attr']  # sensitive attribute
    df_val = df[['prob', 'label', 'pred']]  # prediction and label
    
    ########################################################
    # get the estimated fairness metrics
    fairResult = FairnessMetrics(
        df[f'pred'].to_numpy(),
        df[f'prob'].to_numpy(),
        df['label'].to_numpy(),
        df['sens_attr'].astype(str).to_numpy(),
        previleged_group=privileged_group,add_perf_difference=add_perf_difference)

    ########################################################
    # get bootstrap samples of the fairness metrics
    bootstrap_results = []
    np.random.seed(SEED)

    for i in tqdm(range(bootstrap_n),miniters=bootstrap_n//10):
        df_shuffled = df_val.sample(
            frac=1, replace=True).reset_index(drop=True)  # shuffle the data (bootstrap with replacement)
        df_shuffled['sens_attr'] = df_g  # add the sensitive attribute
        # get the estimated fairness metrics for the bootstrap sample
        fairResult_bootstrap = FairnessMetrics(
            df_shuffled[f'pred'].to_numpy(),
            df_shuffled[f'prob'].to_numpy(),
            df_shuffled['label'].to_numpy(),
            df_shuffled['sens_attr'].astype(str).to_numpy(),
            previleged_group=privileged_group,add_perf_difference=add_perf_difference)

        bootstrap_results.append(fairResult_bootstrap)
    df_bootstrap = pd.DataFrame.from_records(bootstrap_results)[METRIC_NAMES_DICT['fairness_metrics']]
    ########################################################
    # get the percentile of the measured values over bootstraped samples
    # p-value of the fairness metrics (biased against the unprevileged group)
    p_table_worse = {}
    # p-value of the fairness metrics (biased in favor of the unprevileged group)
    p_table_better = {}
    for metric in METRIC_NAMES_DICT['fairness_metrics']:
        measure = fairResult[metric]
        n_valid_bootstrap = df_bootstrap[metric].notna().sum()
        if n_valid_bootstrap < bootstrap_n:
            print(
                f'Warning: {n_valid_bootstrap} valid bootstrap samples for {metric}. Expected {bootstrap_n}')
        if metric in METRIC_NAMES_DICT['lower_better_fairness_metrics']:
            p_table_worse[metric] = (df_bootstrap[metric] >=
                                     measure).sum() / n_valid_bootstrap
            p_table_better[metric] = (df_bootstrap[metric] <=
                                      measure).sum() / n_valid_bootstrap
        else:
            p_table_worse[metric] = (df_bootstrap[metric] <=
                                     measure).sum() / n_valid_bootstrap
            p_table_better[metric] = (df_bootstrap[metric] >=
                                      measure).sum() / n_valid_bootstrap
    p_table_worse = pd.Series(
        p_table_worse)
    p_table_better = pd.Series(
        p_table_better)
    fairResult = pd.DataFrame(fairResult)
    p_table_worse = pd.DataFrame(p_table_worse, columns=['pval']).transpose()
    p_table_better = pd.DataFrame(p_table_better, columns=['pval']).transpose()

    return fairResult, p_table_worse, p_table_better

def main(
    folder,
    n_bootstrap=1000,
    fix_privileged_group=False,
    aggregate_method: Literal['concatenate','fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george','groupwise'] = 'fisher',
    insert_columns={}):
    '''
    Perform the bias test for the model
    '''
    
    dfs, demo_col = get_results_from_folder(folder)
    ## define the privileged group    
    PRIVILEGED_GROUP_DICT = {
        'gender': 'MALE',
        'race': "WHITE",
        'age': 'below'
    }
    if fix_privileged_group:
        privileged_group = PRIVILEGED_GROUP_DICT[demo_col]

        if demo_col == 'age':
            # if the demographic attribute is age, we use the median age as the threshold
            # find median age
            age = [df['sens_attr'] for df in dfs] 
            age = pd.concat(age)
            median_age = age.median()
            for i in range(len(dfs)):
                dfs[i]['sens_attr'] = [f'below{median_age:.1f}' if i < median_age else f'above{median_age:.1f}' for i in dfs[i]['sens_attr']]

            privileged_group = f'{PRIVILEGED_GROUP_DICT["age"]}{median_age:.1f}'

        else:
            # if the demographic attribute is race or gender, we use the predefined privileged group
            privileged_group = PRIVILEGED_GROUP_DICT[demo_col]
    else:
        if demo_col == 'age':
            # if the demographic attribute is age, we use the median age as the threshold
            # find median age
            age = [df['sens_attr'] for df in dfs] 
            age = pd.concat(age)
            median_age = age.median()
            for i in range(len(dfs)):
                dfs[i]['sens_attr'] = [f'below{median_age:.1f}' if i < median_age else f'above{median_age:.1f}' for i in dfs[i]['sens_attr']]

        privileged_group = None 
        
    
    # if the demographic attribute is race, we estimate the p-values for each minority group separately
    if demo_col == 'race':
        race_dfs_dict = {}
        PRIVILIGED_RACE = PRIVILEGED_GROUP_DICT['race']
        for i, min_race in enumerate(MINORITY_RACES):
            # filter the data to include only 2 races
            dfs_race = []
            include_races = [PRIVILIGED_RACE, min_race]
            for df in dfs:
                df_race = df.loc[df['sens_attr'].isin(include_races)].reset_index(drop=True).copy()
                dfs_race.append(df_race)
            min_count = [len(x.loc[x['sens_attr']==min_race]) for x in dfs_race]
            if min(min_count) == 0:
                print(f'Warning: {min_race} has 0 samples in some folds. Skipping this task.')
                continue
            race_dfs_dict[min_race] = dfs_race
        race_dfs_dict['All'] = dfs
        
    if demo_col == 'race':
        ## if the demographic attribute is race, estimate the p-values for each minority group separately
        dfs_p_worse = []
        dfs_p_better = []
        dfs_fairResult = []
        for min_race, dfs_race in race_dfs_dict.items():
            insert_columns_race = insert_columns.copy()
            insert_columns_race['sens_attr'] = f'{demo_col} ({PRIVILIGED_RACE.lower()} vs. {min_race.lower()})'
            df_p_worse, df_p_better, fairResult = CV_bootstrap_bias_test(
                dfs_race, privileged_group=privileged_group, n_bootstrap=n_bootstrap,aggregate_method=aggregate_method)
            for i, (key, val) in enumerate(insert_columns_race.items()):
                df_p_worse.insert(i, key, val)
                df_p_better.insert(i, key, val)
                fairResult.insert(i, key, val)
            dfs_p_worse.append(df_p_worse)
            dfs_p_better.append(df_p_better)
            dfs_fairResult.append(fairResult)
        df_p_worse = pd.concat(dfs_p_worse)
        df_p_better = pd.concat(dfs_p_better)
        fairResult = pd.concat(dfs_fairResult)
    else:
        df_p_worse, df_p_better, fairResult = CV_bootstrap_bias_test(
            dfs, privileged_group=privileged_group, n_bootstrap=n_bootstrap,aggregate_method=aggregate_method)
        ## insert the columns
        for i, (key, val) in enumerate(insert_columns.items()):
            df_p_worse.insert(i, key, val)
            df_p_better.insert(i, key, val)
            fairResult.insert(i, key, val)
            
    return df_p_worse, df_p_better, fairResult

    


if __name__ == '__main__':
    # list all the subfolders in the baseline folders
    subfolders = glob.glob(join(
        args.results_folder, '*ffpe')) + glob.glob(join(args.baseline_folder, '*frozen'))
    subfolder_names = [basename(f) for f in subfolders]
    # find matching folders
    df = pd.DataFrame(
        {'folder': subfolders, 'name': subfolder_names})
    # df.to_csv('TCGA_task_list.csv')
    ##
    print(df)
    for i, row in df.iterrows():
        folder = row['folder']
        proj = row['name']
        ATTR_RENAME_MAP = {'female':'gender','white':'race','age':'age'}  # rename the demographic attribute
        attr = proj.split(' ')[0]
        demo_col = ATTR_RENAME_MAP[attr]
        # define the task
        task = ' '.join(proj.split(' ')[1:-1])
        sample_type = proj.split(' ')[-1]
        
        ###
            
        print(f'Processing {proj} ({i+1}/{len(df)})...')
        insert_columns = {'proj':proj,'task': task, 'sample_type': sample_type,'sens_attr': demo_col,'model':'baseline'}
        df_p_worse, df_p_better, fairResult = main(
            baseline_folder, args.n_bootstrap, aggregate_method=args.aggregate_method,insert_columns=insert_columns,fix_privileged_group=args.fix_privileged_group)

        ## save the p-values
        output_folder = join(args.output_folder,f'agg_{args.aggregate_method}')
        os.makedirs(output_folder, exist_ok=True)
        df_p_better.to_csv(join(output_folder, f'p_better_{proj}({ args.n_bootstrap}samples).csv'))
        df_p_worse.to_csv(join(output_folder, f'p_worse_{proj}({ args.n_bootstrap}samples).csv'))
        fairResult.to_csv(join(output_folder, f'fairResult_{proj}.csv'))

