import argparse
import os
import random
from datetime import datetime
import string
import json


def parser_setting():
    parser = argparse.ArgumentParser(description = 'Classification.')

    ## 
    parser.add_argument('--datasetpath', type = str, nargs = '+', default = ['/home/yen/CancerFairness/Cancer/dataset/white BRCA v0 40 Frozen Positive.pkl'], help = "set datasets file path.")
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
                os.mkdir(args.model_save_directory)
        while 1:
            strRandomDir =  random.choices(string.ascii_letters, k=10)
            strRandomDir = "".join(strRandomDir)
            strSavedWeightDir = f'{args.model_save_directory}/{strRandomDir}'
            if not os.path.exists(strSavedWeightDir) or not os.path.isdir(strSavedWeightDir):
                os.mkdir(strSavedWeightDir)
                print(strSavedWeightDir)
                args.model_save_directory = strSavedWeightDir
                break
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
