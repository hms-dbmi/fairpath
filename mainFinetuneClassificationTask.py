import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairness_aware_classification.metrics import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from argparse import ArgumentParser
from os.path import join, basename, isfile
import pickle
import os
from utils import FairnessMetrics
from bootstrap_bias_test import CV_bootstrap_bias_test
from aif360.sklearn.preprocessing import Reweighing
import warnings
import glob
from tqdm import tqdm
import torch
import json

# --- Constants from convert_dataset_fairpath.py ---
CLINICAL_BASE = 'TCGA_all_clinical'
SEED = 0
SUBJ_ID_COL = 'bcr_patient_barcode'
EXCLUDE_RACES = ['AMERICAN INDIAN OR ALASKA NATIVE', '[not available]']
MINORITY_RACES = ['BLACK OR AFRICAN AMERICAN', 'ASIAN']
N_FOLDS = 4
# SENS_ATTR_MAP is already defined in TCGA_finetune.py

# Select base classfiers for the meta-classifiers
base_clf = LogisticRegression()

SENS_ATTR_MAP = {
    'white': 'race',
    'female': 'gender',
    'low': 'age'
}

# --- Helper functions from convert_dataset_fairpath.py ---


def get_TCGA_demo_info(
        demo_col,
        demo_folder='TCGA_all_clinical'):
    '''
    get the demographic information from the TCGA clinical files
    Inputs:
        demo_col: str, the demographic column to extract (e.g., 'race', 'gender', 'age')
        demo_folder: str, the folder containing the clinical files (can be downloaded from the GDC portal)
    Returns:
        df_demo: pd.DataFrame
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


def load_features(folder):
    pt_files = glob.glob(join(folder, '*.pt'))
    slides = [basename(f).replace('.pt', '') for f in pt_files]
    features = [torch.load(f, map_location='cpu') for f in pt_files]
    features = torch.stack(features, dim=0).numpy()
    return slides, features


def convert_dataset(folder):
    '''
    convert the dataset in the folder to the format used in the experiments
    '''
    sens_attr = basename(folder).split(' ')[0]
    sens_attr = SENS_ATTR_MAP[sens_attr]

    gt_json_file = join(folder, 'groundTruthResults.json')
    assert os.path.isfile(
        gt_json_file), f"ground truth file {gt_json_file} not found"
    with open(gt_json_file) as f:
        gt_json = json.load(f)
    gt_json = [{'slide': key, 'label': val} for key, val in gt_json.items()]
    df_gt = pd.DataFrame(gt_json)
    df_gt[SUBJ_ID_COL] = df_gt['slide'].apply(
        lambda x: '-'.join(x.split('-')[:3]))
    df_demo = get_TCGA_demo_info(demo_col=sens_attr)

    df_gt_demo = df_gt.merge(df_demo, on=SUBJ_ID_COL, how='left')
    df_gt_demo = df_gt_demo.set_index('slide')

    NA_STRS = ['[Not Available]', '[Not Applicable]',
               '[Not Evaluated]', '[Completed]']
    df_gt_demo.replace(NA_STRS, np.nan, inplace=True)
    df_gt_demo = df_gt_demo.dropna(subset=['sens_attr'])
    if sens_attr == 'age':
        df_gt_demo['sens_attr'] = df_gt_demo['sens_attr'].astype(float)/-365.25

    all_folds_data = []
    previous_test_pids = []

    for fold in range(N_FOLDS):
        gt_json_file = join(folder, f'groundTruthResults_test{fold}_0.json')
        with open(gt_json_file) as f:
            gt_json = json.load(f)
        gt_json = {k: v for k, v in gt_json.items(
        ) if k not in previous_test_pids}
        previous_test_pids.extend(list(gt_json.keys()))

        slides, features = load_features(join(folder, f'feature_new{fold}_0'))
        slides_test = list(gt_json.keys())
        slides_test = df_gt_demo.index.intersection(slides_test)

        slides_train = list(set(slides) - set(slides_test))
        slides_train = df_gt_demo.index.intersection(slides_train)

        idx_train = [slides.index(s) for s in slides_train]
        features_train = features[idx_train]
        idx_test = [slides.index(s) for s in slides_test]
        features_test = features[idx_test]

        df_train = df_gt_demo.loc[slides_train].reset_index()
        df_test = df_gt_demo.loc[slides_test].reset_index()

        train_dict = {
            'slide': df_train['slide'].to_list(),
            'bcr_patient_barcode': df_train[SUBJ_ID_COL].to_list(),
            'label': df_train['label'].to_numpy(),
            'sens_attr': df_train['sens_attr'].to_numpy(),
            'features': features_train,
        }

        test_dict = {
            'slide': df_test['slide'].to_list(),
            'bcr_patient_barcode': df_test[SUBJ_ID_COL].to_list(),
            'label': df_test['label'].to_numpy(),
            'sens_attr': df_test['sens_attr'].to_numpy(),
            'features': features_test,
        }
        all_folds_data.append((train_dict, test_dict))
    return all_folds_data


def save_results(outpath, clf, fold, df_results, scores):
    # create the output folder
    os.makedirs(outpath, exist_ok=True)
    # save the results
    df_results.to_csv(join(outpath, f'results_fold{fold}.csv'), index=False)
    # save the scores
    df_score = pd.DataFrame(scores)
    df_score.to_csv(join(outpath, f'scores_fold{fold}.csv'), index=False)
    # save trained model in pickle format
    with open(join(outpath, f'model_fold{fold}.pkl'), 'wb') as f:
        pickle.dump(clf, f)


def check_sufficient_groups(y, group, min_N=8):
    ''' 
    check if the data is sufficient for the classifier
    (for each group, each label should have at least min_N samples)
    y: list of labels
    group: list of groups) 
    returns: group names with sufficient data
    '''
    df = pd.DataFrame({'y': y, 'group': group})
    counts = df.groupby(['y', 'group']).size().reset_index(name='counts')
    counts = counts.groupby('group')['counts'].min()
    check_counts = counts >= min_N

    # check if all groups have two or more labels
    n_classes = df.groupby('group').nunique()['y']
    check_classes = n_classes >= 2
    check_all = check_counts & check_classes
    valid_groups = list(check_all[check_all].index)
    return valid_groups


def run_AUROC_onesample_bootstrap_test(y_true, y_pred, n_bootstraps=1000):
    # y_true: true labels
    # y_pred: predicted probabilities
    # baseline: the baseline value to compare the AUROC to
    # returns: p-value of the one-sample bootstrap test
    auroc = roc_auc_score(y_true, y_pred)
    auc_bootstraps = []
    for i in range(n_bootstraps):
        indices1 = np.random.choice(
            range(len(y_true)), len(y_true), replace=True)
        indices2 = np.random.choice(
            range(len(y_true)), len(y_true), replace=True)
        auc_bootstraps.append(roc_auc_score(
            y_true[indices1], y_pred[indices2]))
    auc_bootstraps = np.array(auc_bootstraps)
    p_value = (np.sum(auc_bootstraps >= auroc)) / (n_bootstraps)
    df = pd.Series({'AUROC': auroc, 'p_value': p_value})  # .to_frame().T
    return df


def save_bootstrap_results(dfs, outpath, n_bootstrap=1000, insert_columns={}):
    # check if the results already exist
    if isfile(join(outpath, f'bootstrapTest_metrics.csv')):
        print(f"Results already exist for bootstrap in {outpath}")
        return
    os.makedirs(outpath, exist_ok=True)

    df_p_worse, df_p_better, fairResult, df_CI = CV_bootstrap_bias_test(
        dfs, privileged_group=None, n_bootstrap=n_bootstrap, aggregate_method='concatenate')

    df_micro = pd.concat(dfs)
    df_auc = run_AUROC_onesample_bootstrap_test(df_micro['label'].to_numpy(
    ), df_micro['prob'].to_numpy(), n_bootstraps=n_bootstrap).reset_index()
    # insert the columns
    for i, (key, val) in enumerate(insert_columns.items()):
        df_p_worse.insert(i, key, val)
        df_p_better.insert(i, key, val)
        fairResult.insert(i, key, val)
        df_CI.insert(i, key, val)
        df_auc.insert(i, key, val)

    postfixs = ['p_AUC', 'p_biased',
                'p_biased_against_majority', 'metrics', 'CI']
    for df, postfix in zip([df_auc, df_p_worse, df_p_better, fairResult, df_CI], postfixs):
        outname = join(outpath, f"bootstrapTest_{postfix}.csv")
        df.to_csv(outname)


def encoder_cancer_type_label(y, project_name):
    '''
    project_name: string (for example, 02_GBM_08_LGG_Classification_race_2025-01-31_21-37-34)
    y: list of labels
    find the order of appreance of the cancer types in the project name (string)
    the labels will be encoded in 0...N by their order of appreance in the project name.
    for example, for project name 02_GBM_08_LGG_Classification_race_2025-01-31_21-37-34
    the GBM will be encoded as 0, LGG as 1
    '''

    unique_labels = list(set(y))
    # find substring in the project name

    first_appreance = [project_name.index(label) for label in unique_labels]
    first_appreance = np.array(first_appreance)
    # sort the labels by their appreance in the string
    sorted_indices = np.argsort(first_appreance)
    sorted_labels = [unique_labels[i] for i in sorted_indices]
    # encode the labels by the order of appreance
    label_encoder = {label: i for i, label in enumerate(sorted_labels)}
    y_encoded = [label_encoder[label]
                 if label in label_encoder else label for label in y]
    y_encoded = np.array(y_encoded)
    return y_encoded


def main(args,  clf_name, clf):
    '''
    clf_name: name of the classifier
    clf: classifier object
    '''
    folder = args.folder
    # decode sensitive attribute
    sens_attr = basename(folder).split(' ')[0]
    sens_attr = SENS_ATTR_MAP[sens_attr]
    #
    dfs_results = []
    outpath = join(args.outpath, basename(folder), clf_name)
    os.makedirs(outpath, exist_ok=True)
    # check if the results already exist
    if isfile(join(outpath, 'bootstrapTest_metrics.csv')) and args.skip_existing:
        print(f"Results already exist for {clf_name} in {outpath}")
        return

    all_folds_data = convert_dataset(folder)

    for fold, (train_npz_data, test_npz_data) in enumerate(all_folds_data):
        print(f"Processing fold {fold}")

        X_train = train_npz_data['features']
        y_train = train_npz_data['label']
        s_train = train_npz_data['sens_attr']
        slide_train = train_npz_data['slide']

        X_test = test_npz_data['features']
        y_test = test_npz_data['label']
        s_test = test_npz_data['sens_attr']
        slide_test = test_npz_data['slide']

        NA_STRS = ['[Not Available]', '[Not Applicable]',
                   '[Not Evaluated]', '[Completed]']

        if sens_attr == 'age':
            s_train = [x if x not in NA_STRS else np.nan for x in s_train]
            s_test = [x if x not in NA_STRS else np.nan for x in s_test]
            s_train = np.array([float(x)/-365.25 for x in s_train])
            s_test = np.array([float(x)/-365.25 for x in s_test])
            median_age = np.nanmedian(np.concatenate([s_train, s_test]))
            s_train = np.array(
                ['old' if x > median_age else 'young' for x in s_train])
            s_test = np.array(
                ['old' if x > median_age else 'young' for x in s_test])

        y_all = np.concatenate([y_train, y_test])
        group_all = np.concatenate([s_train, s_test])
        # check valid groups
        valid_groups = check_sufficient_groups(y_all, group_all)
        if len(valid_groups) < 2:
            print(f"Not enough data for {clf_name} in {folder}")
            return
        else:
            # keep only the valid groups
            idx_train = np.isin(s_train, valid_groups)
            idx_test = np.isin(s_test, valid_groups)
            s_train = s_train[idx_train]
            s_test = s_test[idx_test]
            y_train = y_train[idx_train]
            y_test = y_test[idx_test]
            X_train = X_train[idx_train]
            X_test = X_test[idx_test]
            slide_train = slide_train[idx_train]
            slide_test = slide_test[idx_test]
            print(
                f"Number of samples in each group X class for {clf_name} in {folder}")

        if isinstance(y_train[0], str):
            y_train = encoder_cancer_type_label(y_train, basename(folder))
            y_test = encoder_cancer_type_label(y_test, basename(folder))

        # encode the sensitive attribute
        le = LabelEncoder()
        s_train_encoded = le.fit_transform(s_train)
        s_test_encoded = le.transform(s_test)

        # remove nans
        idx_train = ~np.isnan(y_train) & ~np.isnan(s_train_encoded)
        print(f"removing {np.sum(~idx_train)} samples from train set")
        X_train = X_train[idx_train]
        y_train = y_train[idx_train]
        s_train = s_train[idx_train]
        s_train_encoded = s_train_encoded[idx_train]

        idx_test = ~np.isnan(y_test) & ~np.isnan(s_test_encoded)
        print(f"removing {np.sum(~idx_test)} samples from test set")
        X_test = X_test[idx_test]
        y_test = y_test[idx_test]
        s_test = s_test[idx_test]
        s_test_encoded = s_test_encoded[idx_test]
        # check if the data is sufficient
        df_train = pd.DataFrame({'y': y_train, 's': s_train})
        df_test = pd.DataFrame({'y': y_test, 's': s_test})
        print(
            f"Number of samples in each group X class for {clf_name} in {folder}")
        print(df_train.groupby(['s', 'y']).size())
        print(df_test.groupby(['s', 'y']).size())
        # apply preprocessing
        rw = Reweighing(prot_attr=s_train_encoded)
        df_Xtrain = pd.DataFrame(X_train)
        df_Xtest = pd.DataFrame(X_test)
        _, w_train = rw.fit_transform(df_Xtrain, y_train)

        # Fit the classifier
        clf.fit(X_train, y_train, sample_weight=w_train)

        # Make predictions and compute scores
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        # compile the results
        results = {
            'slide': slide_test,
            'sens_attr': s_test,
            'label': y_test,
            'pred': y_pred,
            'prob': y_prob,
        }
        df_results = pd.DataFrame(results)
        dfs_results.append(df_results)
        scores = FairnessMetrics(
            y_pred,
            y_prob,
            y_test,
            s_test, add_perf_difference=True)

        save_results(outpath, clf, fold, df_results, scores)
        print(clf_name + ' finished')

    # print the number of case per group X class
    df = pd.concat(dfs_results)
    print(
        f"Number of samples in each group X class for {clf_name} in {folder}")
    print(df[['label', 'sens_attr']].value_counts().sort_index())
    # perform bootstrap test
    insert_columns = {
        'task': basename(folder),
        'sens_attr': basename(folder).split(' ')[0],
        'sample_type': basename(folder).split(' ')[-1]}

    # ignore the warning
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    # save the bootstrap results
    save_bootstrap_results(
        dfs_results, outpath, n_bootstrap=args.n_bootstrap, insert_columns=insert_columns)


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        '--folder', type=str, default='TCGA_CONVERTED_DATASET')
    parser.add_argument('--outpath', type=str, default='results')
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--skip_existing', action='store_true', default=False)
    args = parser.parse_args()
    # print args
    print("====== Arguments ======")
    [print(f"{k}: {v}") for k, v in vars(args).items()]
    print("=======================")

    clf_name = 'Reweighting'
    clf = base_clf
    print(f"Processing {clf_name} for folder {args.folder}")
    # Load the data
    # Create the output folder
    main(args, clf_name, clf)
