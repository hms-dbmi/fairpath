import argparse
import os
import random
from datetime import datetime
import string
import json
import numpy as np
from util import flatten_nested_list
from sklearn.model_selection import train_test_split

def parser_setting():
    parser = argparse.ArgumentParser(description = 'Classification.')

    ##
    parser.add_argument('--datasetpath', type = str, nargs = '+', default = ['/home/yen/CancerFairness/Cancer/dataset/white BRCA v0 40 Frozen Positive.pkl'], help = "set datasets file path.")
    parser.add_argument('--projectname', type = str, default = None, help = "set project name.")
    parser.add_argument('--patchesdirectory', type = str, nargs = '+', default = ['/home/yen/nas/TCGA/Patches/1_BRCA/'], help = "set patches path.")
    parser.add_argument('--patchesinformation', type = str, nargs = '+', default = ['/home/yen/CancerFairness/Statistics/01_BRCA/img_information.pkl'])

    parser.add_argument('--testpath', type = str, nargs = '+', default = [], help = "set test datasets file path.")
    parser.add_argument('--testdirectory', type = str, nargs = '+', default = [], help = "set test patches path.")

    parser.add_argument('--model_save_directory', type = str, default = '/home/yen/CancerFairness/Cancer/01_BRCA/weight/')
    parser.add_argument('--pretraineddirectory', type = str, default = '')
    parser.add_argument("--learning_rate", type = float, default = 5e-4, help = "set the learning rate.")
    parser.add_argument("--epoch", type = int, default = 50, help = "set the epoch.")
    parser.add_argument("--step", type = int, default = 50, help = "set the step.")

    parser.add_argument("--optimizer", type = str, default = "SGD", help = "set the optimizer.")
    parser.add_argument("--cpu", action = 'store_true', help = "set only cpu.")
    parser.add_argument("--batch_size", type = int, default = 2, help = "set the batch size.")
    parser.add_argument("--number_of_workers", type = int, default = 8, help = "set the number of workers.")
    parser.add_argument('--balanced', action = 'store_true', help = "set data balanced.")

    parser.add_argument("--outerloop", type = int, default = 4)
    parser.add_argument("--innerloop", type = int, default = 1)
    parser.add_argument("--specificInnerloop", type = int, default= -1)
    parser.add_argument("--pretrainedepoch", type = str, default = '200')

    parser.add_argument('--wandb', action = 'store_true', help = "set wandb.")
    parser.add_argument('--wandb_projectname', type = str, default = '', help = "set wandb project name.")

    parser.add_argument('--number_of_class', type = int, default = 2, help = "set number of class.")

    parser.add_argument('--pickType', type = str, default = 'test', help = "set patches picking way.")
    parser.add_argument('--multiply', type = int, default = 24, help = "set multiply ratio.")
    parser.add_argument('--noise', action = 'store_true', help = "set noise.")

    parser.add_argument('--imbalance', type = str, default = '', help = "introduce either group or class")
    parser.add_argument('--group_imbalance_ratio', type = int, nargs = '+', default = [200, 0], help = "amount of two different group's input data. [group0, group1]")
    parser.add_argument('--class_imbalance_ratio', type = int, nargs = '+', default = [200, 0, 0, 200], help = "amount of two different class's input data for group0 and group1. [group0class0, group0class1, group1class0, group1class1]")


    args = parser.parse_args()

    if len(args.patchesdirectory) != len(args.datasetpath):
        raise ValueError(f'Length of patchesdirectory:{len(args.patchesdirectory)} is not as same as datasetpath:{len(args.datasetpath)}.')
    if args.wandb == True and args.wandb_projectname == '':
        raise ValueError('Please name the project of this experiment.')

    return args

def create_directory(args):
    try:
        random.seed(datetime.now().timestamp())
        if not (os.path.exists(args.model_save_directory) or os.path.isdir(args.model_save_directory)):
                os.makedirs(args.model_save_directory, exist_ok=True)
        if not args.imbalance:
            while 1:
                if args.projectname is not None:
                    strRandomDir = args.projectname
                    strSavedWeightDir = os.path.join(args.model_save_directory, args.projectname)
                else:
                    strRandomDir =  random.choices(string.ascii_letters, k=10)
                    strRandomDir = "".join(strRandomDir)
                    strSavedWeightDir = f'{args.model_save_directory}/{strRandomDir}'
                if not os.path.exists(strSavedWeightDir) or not os.path.isdir(strSavedWeightDir):
                    os.mkdir(strSavedWeightDir)
                    print(strSavedWeightDir)
                    args.model_save_directory = strSavedWeightDir
                    break
        else:
            strRandomDir = args.model_save_directory
    except Exception as e:
        print(f'[Error] Create the model saved directory: {e}.')
        exit()
    return strRandomDir

def save_args(args):
    argsdic = vars(args)
    json_object = json.dumps(argsdic)
    with open(os.path.join(args.model_save_directory, 'args.json'), "w") as outfile:
        outfile.write(json_object)

def dataset_generator(args, IdxOuterLoop, IdxInnerLoop, ListDataset, Last = -1):
    ListTrainSet = []
    tmpSetMag = [1 for _ in range(len(ListDataset))]
    tmpSet = []
    ## add training set in temporary set
    for samples in ListDataset:
        if(Last != -1):
            tmpSet.append(samples['fnlist'][IdxOuterLoop][0][IdxInnerLoop][0][:Last])
        else:
            tmpSet.append(samples['fnlist'][IdxOuterLoop][0][IdxInnerLoop][0][:])
    ## if balanced is True, update tmpSetMag
    if args.balanced == True:
        tmpSetLeng = [len(i) for i in tmpSet]
        tmpSetMag  = [round(max(tmpSetLeng)/i) for i in tmpSetLeng]
    ## follow the mag, add dataset in training set
    for magidx, miniset in enumerate(tmpSet):
        ListTrainSet.append(miniset*tmpSetMag[magidx])
        print(f'Set {magidx} now is added in training set. Length: {len(miniset)*tmpSetMag[magidx]}')

    ListValidSet = [samples['fnlist'][IdxOuterLoop][0][IdxInnerLoop][1][:] for samples in ListDataset]

    return ListTrainSet, ListValidSet

def datasetGenerator(IdxOuterLoop, IdxInnerLoop, ListDataset, balanced = False, Last = -1):
    ListTrainSet = []
    tmpSetMag = [1 for _ in range(len(ListDataset))]
    tmpSet = []
    ## add training set in temporary set
    for samples in ListDataset:
        if(Last != -1):
            tmpSet.append(samples['fnlist'][IdxOuterLoop][0][IdxInnerLoop][0][:Last])
        else:
            tmpSet.append(samples['fnlist'][IdxOuterLoop][0][IdxInnerLoop][0][:])
    ## if balanced is True, update tmpSetMag
    if balanced == True:
        tmpSetLeng = [len(i) for i in tmpSet]
        tmpSetMag  = [round(max(tmpSetLeng)/i) for i in tmpSetLeng]
    ## follow the mag, add dataset in training set
    for magidx, miniset in enumerate(tmpSet):
        ListTrainSet.append(miniset*tmpSetMag[magidx])
        print(f'Set {magidx} now is added in training set. Length: {len(miniset)*tmpSetMag[magidx]}')

    ListValidSet = [samples['fnlist'][IdxOuterLoop][0][IdxInnerLoop][1][:] for samples in ListDataset]

    return ListTrainSet, ListValidSet

def imbalanced_dataset_generator(args, ListDataset):
    # male = 1, female = 0
    # black = 1, white = 0
    # high = 1, low = 0
    finalSet = []
    for sample in ListDataset:
        tmpSet = []
        for f in range(1):
            fold = sample['fnlist'][f]
            flattened_list = flatten_nested_list(fold)
            for i in range(0, len(flattened_list), 4):
                sublst = flattened_list[i:i + 4]
                if sublst not in tmpSet:
                    tmpSet.append(sublst)
            # finalSet's structure: [[[x, x, x, x]...], [[x, x, x, x]...], [[x, x, x, x]...], [[x, x, x, x]...]]
            # g0c0, g0c1, g1c0, g1c1
            finalSet.append(tmpSet)

    print(f"length of group0class0: {len(finalSet[0])}, "
          f"group0class1: {len(finalSet[1])}, "
          f"group1class0: {len(finalSet[2])}, "
          f"group1class1: {len(finalSet[3])}, ")

    if args.imbalance == "group":
        concatTrainSet = [[], []] # [[g0], [g1]]

    if args.imbalance == "class":
        concatTrainSet = [[], [], [], []] # [[g0c0], [g0c1], [g1c0], [g1c1]]

    ListTestSet = []
    for idx, samples in enumerate(finalSet):
        # Split the data into training set (60%) and temp set (40%)
        train_data, test_data = train_test_split(samples, test_size=0.3, random_state=42)
        ListTestSet.append(test_data)

        if args.imbalance == "group":
            if idx in [0, 1]:
                # maxture of group0class0 and group0class1
                concatTrainSet[0] += train_data
            else:
                # maxture of group1class0 and group1class1
                concatTrainSet[1] += train_data

        if args.imbalance == "class":
            concatTrainSet[idx] += train_data

    if args.imbalance == "group":
        print(f"Group0TrainSet: {len(concatTrainSet[0])}, Group1TrainSet: {len(concatTrainSet[1])}")

    if args.imbalance == "class":
        print(f"Group0Class0TrainSet: {len(concatTrainSet[0])}, "
              f"Group0Class1TrainSet: {len(concatTrainSet[1])}, "
              f"Group1Class0TrainSet: {len(concatTrainSet[2])}, "
              f"Group1Class1TrainSet: {len(concatTrainSet[3])}")
    # if args.balanced == True:
    #     tmpMag = round(max(len(Group0ListTrainSet), len(Group1ListTrainSet))/min(len(Group0ListTrainSet), len(Group1ListTrainSet)))

    #     if len(Group0ListTrainSet) < len(Group1ListTrainSet):
    #         Group0ListTrainSet *= tmpMag
    #     else:
    #         Group1ListTrainSet *= tmpMag
    if args.imbalance == "group":
        randomG0TrainIdxs = np.random.choice(len(concatTrainSet[0]), args.group_imbalance_ratio[0], replace=False)
        randomG0TrainSet = np.array(concatTrainSet[0])[randomG0TrainIdxs]
        randomG1TrainIdxs = np.random.choice(len(concatTrainSet[1]), args.group_imbalance_ratio[1], replace=False)
        randomG1TrainSet = np.array(concatTrainSet[1])[randomG1TrainIdxs]
        print(f"Length of randomG0TrainSet: {len(randomG0TrainSet)}, randomG1TrainSet: {len(randomG1TrainSet)}")
        ListTrainSet = split_group_n_class(randomG0TrainSet.tolist(), randomG1TrainSet.tolist())

    if args.imbalance == "class":
        randomG0C0TrainIdxs = np.random.choice(len(concatTrainSet[0]), args.class_imbalance_ratio[0], replace=False)
        randomG0C0TrainSet = np.array(concatTrainSet[0])[randomG0C0TrainIdxs]
        randomG0C1TrainIdxs = np.random.choice(len(concatTrainSet[1]), args.class_imbalance_ratio[1], replace=False)
        randomG0C1TrainSet = np.array(concatTrainSet[1])[randomG0C1TrainIdxs]
        randomG1C0TrainIdxs = np.random.choice(len(concatTrainSet[2]), args.class_imbalance_ratio[2], replace=False)
        randomG1C0TrainSet = np.array(concatTrainSet[2])[randomG1C0TrainIdxs]
        randomG1C1TrainIdxs = np.random.choice(len(concatTrainSet[3]), args.class_imbalance_ratio[3], replace=False)
        randomG1C1TrainSet = np.array(concatTrainSet[3])[randomG1C1TrainIdxs]
        print(f"Length of"
              f"randomG0C0TrainSet: {len(randomG0C0TrainSet)}, "
              f"randomG0C1TrainSet: {len(randomG0C1TrainSet)}, "
              f"randomG1C0TrainSet: {len(randomG1C0TrainSet)}, "
              f"randomG1C1TrainSet: {len(randomG1C1TrainSet)}, ")
        ListTrainSet = [randomG0C0TrainSet.tolist(), randomG0C1TrainSet.tolist(), randomG1C0TrainSet.tolist(), randomG1C1TrainSet.tolist()]

    pos_weight = 1 / ((len(ListTrainSet[1]) + len(ListTrainSet[3])) / (len(ListTrainSet[0]) + len(ListTrainSet[2])))

    print(
        f"len of Train set g0c0: {len(ListTrainSet[0])}\n"
        f"len of Train set g0c1: {len(ListTrainSet[1])}\n"
        f"len of Train set g1c0: {len(ListTrainSet[2])}\n"
        f"len of Train set g1c1: {len(ListTrainSet[3])}\n"
        f"len of test set g0c0: {len(ListTestSet[0])}\n"
        f"len of test set g0c1: {len(ListTestSet[1])}\n"
        f"len of test set g1c0: {len(ListTestSet[2])}\n"
        f"len of test set g1c1: {len(ListTestSet[3])}\n"
        f"positive weight: {pos_weight}\n"
    )
    return ListTrainSet, ListTestSet, pos_weight

def split_group_n_class(*groups):
    ListTrainSet = [[], [], [], []]
    for idx, group in enumerate(groups):
        for data in group:
            # when class is 0
            if data[0] == '0':
                ListTrainSet[0 if idx == 0 else 2].append(data)
            # when class is 1
            else:
                ListTrainSet[1 if idx == 0 else 3].append(data)
    return ListTrainSet
