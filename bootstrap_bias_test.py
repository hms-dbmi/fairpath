'''
Script to calculate the bias of models
'''

import pandas as pd
import numpy as np
from scipy.stats import combine_pvalues
from fairness_utils import FairnessMetrics, get_metric_names
from tqdm import tqdm


SEED = 0
METRIC_NAMES_DICT = get_metric_names(add_perf_difference=True)


def CV_bootstrap_bias_test(
        dfs, privileged_group=None, n_bootstrap=1000, aggregate_method='fisher'):
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
        fairResult, df_p_worse, df_p_better, _ = bootstrap_bias_test(df,
                                                                     bootstrap_n=n_bootstrap, privileged_group=privileged_group)
        df_CI, _ = bootstrap_bias_CI(
            df, bootstrap_n=n_bootstrap, privileged_group=privileged_group)

        return df_p_worse, df_p_better, fairResult, df_CI

    if aggregate_method == 'fold_micro':
        # perform bootstrap withing each fold, but test on the micro-averaged results
        for i in range(len(dfs)):
            dfs[i]['fold'] = i
        df = pd.concat(dfs).reset_index(drop=True)
        fairResult, df_p_worse, df_p_better, _ = bootstrap_bias_test(df,
                                                                     bootstrap_n=n_bootstrap, privileged_group=privileged_group)
        df_CI, _ = bootstrap_bias_CI(
            df, bootstrap_n=n_bootstrap, privileged_group=privileged_group)

        return df_p_worse, df_p_better, fairResult, df_CI
    elif aggregate_method in ['fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george']:
        # if the method is fisher or stouffer, we calculate the p-value for each fold
        dfs_p_better = []
        dfs_p_worse = []
        fairResult_list = []
        dfs_CI = []
        for i, df in enumerate(dfs):
            fairResult, df_p_worse, df_p_better, _ = bootstrap_bias_test(df,
                                                                         bootstrap_n=n_bootstrap, privileged_group=privileged_group)

            df_CI, _ = bootstrap_bias_CI(
                df, bootstrap_n=n_bootstrap, privileged_group=privileged_group)
            df_p_better.insert(0, 'fold', i)
            df_p_worse.insert(0, 'fold', i)
            fairResult.insert(0, 'fold', i)
            dfs_CI.insert(0, df_CI)
            dfs_p_better.append(df_p_better)
            dfs_p_worse.append(df_p_worse)
            fairResult_list.append(fairResult)
            dfs_CI.append(df_CI)
        # concatenate the p-values
        df_p_better = pd.concat(dfs_p_better)
        df_p_worse = pd.concat(dfs_p_worse)
        fairResult = pd.concat(fairResult_list)
        df_CI = pd.concat(dfs_CI)
        # aggregate the p-values
        df_p_combined = dfs_p_better[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_better[col].loc[i]
                meta_res = combine_pvalues(
                    pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_better = pd.concat([df_p_better, df_p_combined])

        df_p_combined = dfs_p_worse[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_worse[col].loc[i]
                meta_res = combine_pvalues(
                    pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_worse = pd.concat([df_p_worse, df_p_combined])
        return df_p_worse, df_p_better, fairResult, df_CI
    elif aggregate_method == 'fold':
        # bootstrap for significance test
        dfs_p_worse = []
        fairResult_list = []
        dfs_CI = []
        dfs_bootstrap = []
        dfs_bootstrap_CI = []
        for i, df in enumerate(dfs):
            # get the actual estimated fairness metrics for each fold

            fairResult = FairnessMetrics(
                df[f'pred'].to_numpy(),
                df[f'prob'].to_numpy(),
                df['label'].to_numpy(),
                df['sens_attr'].astype(str).to_numpy(),
                previleged_group=privileged_group, add_perf_difference=True)
            fairResult = pd.DataFrame(fairResult)
            fairResult_list.append(fairResult)

            # get the bootstrap samples for each fold

            _, _, _, df_bootstrap = bootstrap_bias_test(df,
                                                        bootstrap_n=n_bootstrap, privileged_group=privileged_group)
            _, df_bootstrap_CI = bootstrap_bias_CI(
                df, bootstrap_n=n_bootstrap, privileged_group=privileged_group)

            dfs_bootstrap.append(df_bootstrap.reset_index())
            dfs_bootstrap_CI.append(df_bootstrap_CI.reset_index())
        # get p-value from bootstrap samples
        # compile the fold results
        df_bootstrap = pd.concat(dfs_bootstrap, axis=0)
        df_bootstrap = df_bootstrap.groupby('index').mean()
        df_bootstrap_CI = pd.concat(dfs_bootstrap_CI, axis=0)
        df_bootstrap_CI = df_bootstrap_CI.groupby('index').mean()
        fairResult = pd.concat(fairResult_list, axis=0)
        fairResult = fairResult.set_index(
            ['sensitiveAttr', 'group_type'], drop=True).mean().to_frame().T
        # fairResult = fairResult.T
        # 'sensitiveAttr': uniSens,
        # 'group_type': group_types,

        ########################################################
        # get the percentile of the measured values over bootstraped samples
        # p-value of the fairness metrics (biased against the unprevileged group)
        p_table_worse = {}
        # p-value of the fairness metrics (biased in favor of the unprevileged group)
        p_table_better = {}
        for metric in METRIC_NAMES_DICT['fairness_metrics']:
            measure = fairResult[metric].values[0]
            n_valid_bootstrap = df_bootstrap[metric].notna().sum()
            if n_valid_bootstrap < n_bootstrap:
                print(
                    f'Warning: {n_valid_bootstrap} valid bootstrap samples for {metric}. Expected {n_bootstrap}')
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
        df_p_worse = pd.DataFrame(p_table_worse, columns=['pval']).transpose()
        df_p_better = pd.DataFrame(
            p_table_better, columns=['pval']).transpose()

        ########################################################
        # get CIs from bootstrap samples

        df_lb = df_bootstrap.quantile(0.025)
        df_ub = df_bootstrap.quantile(0.975)
        df_lb.name = 'CI_lb'
        df_ub.name = 'CI_ub'
        df_CI = pd.concat([df_lb, df_ub], axis=1).T

        return df_p_worse, df_p_better, fairResult, df_CI

    elif aggregate_method == 'groupwise':
        # if the method is groupwise, we estimate the fairness metrics first, and then perform bootstraping on population level
        df_perf_avgdiff, df_p_worse, df_p_better, df_CI = bootstrap_bias_test_groupLevel(dfs,
                                                                                         bootstrap_n=n_bootstrap, privileged_group=privileged_group)

        return df_p_worse, df_p_better, df_perf_avgdiff, df_CI


def get_mean_perf_diff(df_perf):
    dfs_group = {group: df_perf.groupby(['group_type']).get_group(
        group) for group in ['majority', 'minority']}
    df_diff = dfs_group['majority'][METRIC_NAMES_DICT['perf_metrics']
                                    ] - dfs_group['minority'][METRIC_NAMES_DICT['perf_metrics']]
    # for metrics that are lower the better, we reverse the sign
    lower_better_perf = list(
        set(METRIC_NAMES_DICT['lower_better_metrics']).intersection(set(df_diff.columns)))
    # df_diff[lower_better_perf] = -df_diff[lower_better_perf]
    return df_diff.mean(), df_diff


def bootstrap_bias_test_groupLevel(dfs,
                                   bootstrap_n=1000, privileged_group=None, add_perf_difference=True):
    '''
    Do the bootstrap test on the group level (bootstrapping on gorup-level performance metrics)
    H0: mean performance difference between the unprivileged group and the privileged group (across all folds) is 0 
    Ha: > 0
    '''
    # get the performance metrics for each fold
    fairResult_list = []
    for i, df in enumerate(dfs):
        fairResult = FairnessMetrics(
            df[f'pred'].to_numpy(),
            df[f'prob'].to_numpy(),
            df['label'].to_numpy(),
            df['sens_attr'].astype(str).to_numpy(),
            previleged_group=privileged_group, add_perf_difference=True)
        fairResult['fold'] = i
        fairResult = pd.DataFrame(fairResult)
        fairResult_list.append(fairResult)
    # df = pd.DataFrame.from_records(fairResult_list)
    df = pd.concat(fairResult_list)
    df_perf = df[['sensitiveAttr', 'group_type', 'fold'] +
                 METRIC_NAMES_DICT['perf_metrics']]
    df_perf = df_perf.set_index(['fold'], drop=True)
    df_perf_groupwise = df_perf.pivot(
        columns='sensitiveAttr').reset_index(drop=True)
    df_perf_groupwise.drop('group_type', axis=1, inplace=True)
    df_perf_mean = df_perf_groupwise.mean(axis=0)
    # get performance difference (positive is biased against the unprivileged group)
    ##
    df_perf_avgdiff, df_perf_diff = get_mean_perf_diff(df_perf)

    # get CI for the performance difference
    dfs_CI = []
    dfs_perf_CI = []
    np.random.seed(SEED)

    for i in tqdm(range(bootstrap_n), miniters=bootstrap_n//10):
        df_shuffled = df_perf_diff.sample(frac=1, replace=True).mean()
        df_perf_shuffled = df_perf_groupwise.sample(
            frac=1, replace=True).mean(axis=0)
        dfs_CI.append(df_shuffled)
        dfs_perf_CI.append(df_perf_shuffled)

    df_bootstrap = pd.concat(dfs_CI, axis=1).T
    df_perf_bootstrap = pd.concat(dfs_perf_CI, axis=1).T
    df_CI = df_bootstrap.quantile([0.025, 0.975])
    # df_CI.index.name = 'CI'
    # df_CI = df_CI.reset_index()
    df_CI.columns = [f'{x}_diff' if x != 'CI' else x for x in df_CI.columns]
    df_CI.loc['CI'] = [[df_CI.loc[0.025, col], df_CI.loc[0.975, col]]
                       for col in df_CI.columns]
    df_CI = df_CI.loc[['CI']].reset_index(drop=True)
    ##
    df_perf_CI = df_perf_bootstrap.quantile([0.025, 0.975])
    row_combined = {}
    for col in df_perf_CI.columns:
        row_combined[col] = [df_perf_CI.loc[0.025, col],
                             df_perf_CI.loc[0.975, col]]
    row_combined = pd.Series(row_combined)
    df_perf_CI.loc['CI'] = row_combined
    df_perf_CI = df_perf_CI.loc[['CI']].reset_index(drop=True)
    df_perf_CI = df_perf_CI.melt()
    df_perf_CI.rename(columns={None: 'metric', 'value': 'CI'}, inplace=True)
    df_perf_CI = df_perf_CI.pivot(
        columns='metric', index='sensitiveAttr', values='CI').reset_index()
    ##
    df_CI = pd.concat([df_CI]*df_perf_CI.shape[0],
                      axis=0).reset_index(drop=True)
    df_CI = df_perf_CI.merge(df_CI, left_index=True, right_index=True)

    # get the bootstrap samples
    bootstrap_results = []
    np.random.seed(SEED)

    for i in tqdm(range(bootstrap_n), miniters=bootstrap_n//10):
        # bootstrap within each fold
        df_shuffled = df_perf.groupby('fold').sample(
            frac=1, replace=True)  # .reset_index(drop=True)
        df_shuffled[['sensitiveAttr', 'group_type']] = df_perf[[
            'sensitiveAttr', 'group_type']]  # add the sensitive attribute
        df_shuffled_avgdiff, _ = get_mean_perf_diff(df_shuffled)
        bootstrap_results.append(df_shuffled_avgdiff)
    df_bootstrap = pd.concat(bootstrap_results, axis=1).T

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
    return fairResult, p_table_worse, p_table_better, df_CI


def bootstrap_bias_test(df,
                        bootstrap_n=1000, privileged_group=None, add_perf_difference=True, verbose=True):
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
    * p_table_worse: pandas Series, p-value of the fairness metrics (biased against the unprevileged group)
    * p_table_better: pandas Series, p-value of the fairness metrics (biased in favor of the unprevileged group)
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
        previleged_group=privileged_group, add_perf_difference=add_perf_difference)
    if verbose:
        df_temp = pd.DataFrame(fairResult)
        df_perf_temp = pd.DataFrame(df_temp[METRIC_NAMES_DICT['perf_metrics']])
        df_fair_temp = pd.DataFrame(
            df_temp[METRIC_NAMES_DICT['fairness_metrics']])
        pd.set_option('display.max_rows', df_temp.shape[1])
        pd.set_option('display.precision', 3)
        df_disp = pd.concat([df_perf_temp.T.reset_index(),
                            df_fair_temp.T.reset_index()], axis=1)
        print('='*75)
        print(f'\t\tEstimated performance & fairness metrics')
        print('='*75)
        print(df_disp)
        print('='*75)

    ########################################################
    # get bootstrap samples of the fairness metrics
    bootstrap_results = []
    np.random.seed(SEED)

    def get_bootstrap_bias_sample(df_val, df_g):
        df_shuffled = df_val.sample(
            # shuffle the data (bootstrap with replacement)
            frac=1, replace=True).reset_index(drop=True)
        df_shuffled['sens_attr'] = df_g  # add the sensitive attribute
        # get the estimated fairness metrics for the bootstrap sample
        fairResult_bootstrap = FairnessMetrics(
            df_shuffled[f'pred'].to_numpy(),
            df_shuffled[f'prob'].to_numpy(),
            df_shuffled['label'].to_numpy(),
            df_shuffled['sens_attr'].astype(str).to_numpy(),
            previleged_group=privileged_group, add_perf_difference=add_perf_difference)
        return fairResult_bootstrap

    for i in tqdm(range(bootstrap_n), miniters=bootstrap_n//10):
        df_shuffled = df_val.sample(
            # shuffle the data (bootstrap with replacement)
            frac=1, replace=True).reset_index(drop=True)
        df_shuffled['sens_attr'] = df_g  # add the sensitive attribute
        # get the estimated fairness metrics for the bootstrap sample
        fairResult_bootstrap = FairnessMetrics(
            df_shuffled[f'pred'].to_numpy(),
            df_shuffled[f'prob'].to_numpy(),
            df_shuffled['label'].to_numpy(),
            df_shuffled['sens_attr'].astype(str).to_numpy(),
            previleged_group=privileged_group, add_perf_difference=add_perf_difference)
        # fairResult = fairResult

        bootstrap_results.append(fairResult_bootstrap)

    df_bootstrap = pd.DataFrame.from_records(bootstrap_results)[
        METRIC_NAMES_DICT['fairness_metrics']]
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
    return fairResult, p_table_worse, p_table_better, df_bootstrap


def bootstrap_bias_CI(df, bootstrap_n=1000, privileged_group=None, add_perf_difference=True):
    '''
    Estimate the confidence interval of the fairness metrics using bootstrap
    Args:
    * df: pandas DataFrame, data, including the following columns:
        * sens_attr: sensitive attributes
        * prob: model probabilities
        * label: ground truth labels
        * pred:  model predictions
    * bootstrap_n: int, number of bootstrap samples
    * privileged_group: str, previleged group name. If None, the group with the best performance will be used.
    Returns:
    '''

    df = df[['sens_attr', 'prob', 'label', 'pred']]
    df_g = df['sens_attr']  # sensitive attribute
    df_label = df['label']
    df_val = df[['prob', 'label', 'pred']]  # prediction and label

    ########################################################
    # get the estimated fairness metrics
    fairResult = FairnessMetrics(
        df[f'pred'].to_numpy(),
        df[f'prob'].to_numpy(),
        df['label'].to_numpy(),
        df['sens_attr'].astype(str).to_numpy(),
        previleged_group=privileged_group, add_perf_difference=add_perf_difference)

    ########################################################
    # get bootstrap samples of the fairness metrics
    bootstrap_results = []
    np.random.seed(SEED)

    def get_bootstrap_bias_sample(df_val, df_g):

        df_shuffled = df.groupby('sens_attr').apply(
            lambda x: x.sample(frac=1, replace=True)).reset_index(drop=True)

        fairResult_bootstrap = FairnessMetrics(
            df_shuffled[f'pred'].to_numpy(),
            df_shuffled[f'prob'].to_numpy(),
            df_shuffled['label'].to_numpy(),
            df_shuffled['sens_attr'].astype(str).to_numpy(),
            previleged_group=privileged_group, add_perf_difference=add_perf_difference)
        return fairResult_bootstrap

    for i in tqdm(range(bootstrap_n), miniters=bootstrap_n//10):
        df_shuffled = df.groupby('sens_attr').apply(
            lambda x: x.sample(frac=1, replace=True)).reset_index(drop=True)

        fairResult_bootstrap = FairnessMetrics(
            df_shuffled[f'pred'].to_numpy(),
            df_shuffled[f'prob'].to_numpy(),
            df_shuffled['label'].to_numpy(),
            df_shuffled['sens_attr'].astype(str).to_numpy(),
            previleged_group=privileged_group, add_perf_difference=add_perf_difference)
        bootstrap_results.append(fairResult_bootstrap)

    df_bootstrap = pd.DataFrame.from_records(bootstrap_results)[
        METRIC_NAMES_DICT['fairness_metrics']]
    #########################################################
    # get the confidence interval
    df_lb = df_bootstrap.quantile(0.025)
    df_ub = df_bootstrap.quantile(0.975)
    df_lb.name = 'CI_lb'
    df_ub.name = 'CI_ub'
    #########################################################
    # get p-values
    ratio_metrics = ['PQD', 'PQD(class)', 'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)',
                     'EOM(Positive)', 'EOM(Negative)', 'AUCRatio']
    diff_metrics = ['EOpp0', 'EOpp1', 'EBAcc',
                    'EOdd', 'EOddAbs', 'EOddMax', 'AUCDiff', 'TOTALACCDIF', 'ACCDIF', 'AUC_diff', 'ACC_diff',
                    'TPR_diff', 'TNR_diff', 'PPV_diff', 'NPV_diff', 'PR_diff', 'NR_diff',
                    'BAcc_diff', 'FPR_diff', 'FNR_diff', 'OverAllAcc_diff',
                    'OverAllAUC_diff', 'TOTALACC_diff']
    ratio_metrics_valid = [x for x in ratio_metrics if x in df.columns]
    diff_metrics_valid = [x for x in diff_metrics if x in df.columns]
    df_ratio_bootstrap = df_bootstrap[ratio_metrics_valid]
    df_diff_bootstrap = df_bootstrap[diff_metrics_valid]

    df_CI = pd.concat([df_lb, df_ub], axis=1).T
    return df_CI, df_bootstrap
