### This finetune with BN
## system library
import os, sys, glob

## data library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

## torch, sklearn library
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

## warmup scheduler
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

## wandb
import wandb

## written library
from networks import ClfNet, RepNet
from datasets import PathologyDataset
from util import Find_Optimal_Cutoff, FairnessMetrics
import util
from framework import parser_setting

## mass library
import random
from datetime import datetime
import string
from mlxtend.plotting import plot_confusion_matrix

PatchesPerBag = 200

def run(dataloader, epo, epoch, model, criterion = None, optimizer = None,  device = 'gpu'):
    if model.training == True: ## Here for training
        color = 'green'
    else: ## Here for validation
        color = 'red'

    lenSamples = 0
    totalLoss = 0
    allPredictions = np.array([])
    allLabels = np.array([])
    pbar = tqdm(enumerate(dataloader), colour = color, total = len(dataloader))
    for _, data in pbar:
        images, labels, _, _, _ =  data
        images = images.to(device)
        ## method 1
        labels = labels.type(torch.float32).to(device)
        
        ##
        if model.training == True:
            optimizer.zero_grad()
            predictions = model(images)
            predictions = predictions.sigmoid().view(-1)

        else:
            with torch.no_grad():
                predictions = model(images)

                # listPredictions = nn.functional.softmax(predictions, dim = 1).clone().detach().cpu().numpy()[:, 1]
                predictions = predictions.sigmoid().view(-1)
                listPredictions = predictions.detach().cpu().numpy()

                # listPredictions = torch.argmax(predictions, dim = 1).clone().detach().cpu().numpy()
                allPredictions = np.append(allPredictions, listPredictions)
                allLabels = np.append(allLabels, labels.detach().to("cpu").numpy())
        ## calculate loss
        if criterion != None:
            loss = criterion(predictions, labels)
            lenSamples += len(predictions)
            totalLoss += loss.clone().detach().cpu().numpy()*len(predictions)
        
        ## calculate grad and update
        if model.training == True:
            loss.backward()
            optimizer.step()       
        
        ## set pbar description
        if model.training == True:
            strTraining = "Train"
        else:
            strTraining = "Valid"
        if criterion != None:
            pbar.set_description(
                f'{strTraining} Iter: {epo+1:03}/{epoch:03}  '
                f'Loss: {loss:3.4f}'
            ) 
        else:
            pbar.set_description(
                f'{strTraining} Iter: {epo+1:03}/{epoch:03}  '
            )
        pbar.update()

    if model.training == False:
        _, _, auc, threshold = Find_Optimal_Cutoff(allLabels.tolist(), allPredictions.tolist())
        after_predictions = torch.ge(torch.tensor(allPredictions), threshold).int()
        same_predictions = torch.eq(after_predictions, torch.tensor(allLabels)).int().sum().item()
        acc = same_predictions/len(allLabels)

    if model.training == True:
        return totalLoss/lenSamples
    else:
        # method 1
        if criterion != None:
            return totalLoss/lenSamples, threshold, acc, auc
        else:
            return  threshold, acc, auc

def mainTrainValid(ListDataset, pklPatchesInformation, device = 'cuda', strWanbdGroupName = ''):
       
    ### k-fold
    for IdxOuterLoop in range(args.outerloop):
        for IdxInnerLoop in range(args.innerloop):
            if args.specificInnerloop != -1 and args.innerloop == 1:
                IdxInnerLoop = args.specificInnerloop
            try:
                if args.wandb == True:
                    wandb.init(project = args.wandb_projectname, group = f'{strWanbdGroupName}', name = f'{IdxOuterLoop}_{IdxInnerLoop}')
                    wandb.define_metric("train_loss")
                    wandb.define_metric("valid_loss")
                    wandb.define_metric("valid_auc")
                    wandb.define_metric("valid_acc")
            except Exception as e:
                print(f'[Error] Wandb init: {e}.')
                exit()

            try:
                strInnerDir = os.path.join(args.model_save_directory, f'{IdxOuterLoop}_{IdxInnerLoop}')
                os.mkdir(strInnerDir)
            except Exception as e:
                print(f'[Error] Folder Created Error: {e}')
                exit()
            
            ListTrainSet = []
            tmpSetMag = [1 for _ in range(len(ListDataset))]
            tmpSet = []
            ## add training set in temporary set
            for samples in ListDataset:
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

            trainDataset = PathologyDataset(
                boxes = ListTrainSet,
                patchesDirectory = args.patchesdirectory,
                pklPatchesInformation = pklPatchesInformation,
                patchesPerBag = PatchesPerBag,
                size = 224,
                pickType = args.pickType,
                transform = util.image_transform(),
                mergeDataset = True
            )
            validDataset = PathologyDataset(
                boxes = ListValidSet,
                patchesDirectory = args.patchesdirectory,
                pklPatchesInformation = pklPatchesInformation,
                patchesPerBag = PatchesPerBag,
                size = 224,
                pickType = 'test',
                mergeDataset = True
            )

            trainLoader = DataLoader(dataset = trainDataset, batch_size = args.batch_size, shuffle = True, num_workers = args.number_of_workers, pin_memory = False)
            validLoader = DataLoader(dataset = validDataset, batch_size = 1, shuffle = False, num_workers = args.number_of_workers, pin_memory = False)
            
            backbone = RepNet(pretrained = True, projectTytpe = 'mlp', feature_normalized = False)

            if args.pretraineddirectory != '':
                print(backbone.load_state_dict(torch.load(f'{args.pretraineddirectory}/{IdxOuterLoop}_{IdxInnerLoop}/weight100.pth'), strict=False))
                print(f'Model Loaded {args.pretraineddirectory}/{IdxOuterLoop}_{IdxInnerLoop}/weight100.pth.')
            else:
                raise ValueError('This is finetune mode. Please check the pretrained weight.')
            

            model = ClfNet()
            model.featureExtractor = backbone.encoder

            for param in model.featureExtractor.parameters():
                param.requires_grad = False
            
            ## model featureExtractor params frozen
            ## other params trained 

            torch.save(model.state_dict(), os.path.join(strInnerDir, 'weight.pth'))

            model.to(device)
            optimizer = optim.SGD(model.classifier.parameters(), lr = args.learning_rate, momentum = 0.9)

            warm_up_epoches = 1
            scheduler_steplr = CosineAnnealingLR(optimizer, args.epoch)
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier = 1, total_epoch = warm_up_epoches, after_scheduler = scheduler_steplr)
            
            criterion = nn.BCELoss()

            ####################### Main Stage #########################
            bestAUC = 0
            bestThreshold = 0
            # trainingLRs = []
            # lenLRs = len(optimizer.param_groups)
            # for i in range(lenLRs):
            #     trainingLRs.append([])

            ## For scheduler ##
            optimizer.zero_grad()
            optimizer.step()
            for epo in range(args.epoch):
                ### Trainin phase
                model.train()
                scheduler_warmup.step()
                trainLoss = run(trainLoader, epo, args.epoch, model, criterion, optimizer, device = device)
                # for idx in range(lenLRs):
                #     trainingLRs[idx].append(optimizer.param_groups[idx]['lr'])

                ### validation phase
                model.eval()
                ## method 1
                validLoss, threshold, acc, auc = run(validLoader, epo, args.epoch, model, criterion, device = device)

                ## method 1
                if bestAUC < auc:
                    bestThreshold = threshold
                    bestAUC = auc
                    torch.save(model.state_dict(), os.path.join(strInnerDir, 'weight.pth'))
                    print('Update AUC model ! ', end = '')

                print(f"AUC: {auc} ACC: {acc} Threshold: {threshold}")
                try:
                    if args.wandb == True:
                        ## method 1
                        wandb.log({
                            "train_loss": trainLoss, 
                            "valid_loss": validLoss, 
                            "valid_auc": auc, 
                            "valid_acc": acc,
                        })
                except Exception as e:
                    print(f'[Error] Wandb record: {e}')
                    exit()

            if args.wandb == True:
                wandb.finish() 

            ################ Write Back ################
            ## method 1
            result = {
                'AUC': bestAUC,
                'threshold': bestThreshold
            }
            json_object = json.dumps(result)
            with open(os.path.join(strInnerDir, 'threshold.json'), "w") as outfile:
                outfile.write(json_object)

    return strInnerDir

def mainTest(ListDataset, pklPatchesInformation, device):

    gtResults = {}
    predResults = {}
    sensitiveResults = {}
    for IdxOuterLoop in range(args.outerloop):
        ListTestSet = [samples['fnlist'][IdxOuterLoop][1][:] for samples in ListDataset]
        testDataset = PathologyDataset(
            boxes = ListTestSet,
            patchesDirectory = args.patchesdirectory,
            pklPatchesInformation = pklPatchesInformation,
            patchesPerBag = PatchesPerBag,
            size = 224,
            pickType = 'test',
            transform = None,
            mergeDataset = True
        )
        for IdxInnerLoop in range(args.innerloop):
            if args.specificInnerloop != -1 and args.innerloop == 1:
                IdxInnerLoop = args.specificInnerloop
            try:
                strInnerDir = os.path.join(args.model_save_directory, f'{IdxOuterLoop}_{IdxInnerLoop}')
                model = ClfNet()
                print(f'Loading model: {strInnerDir}')
                print(model.load_state_dict(torch.load(f'{strInnerDir}/weight.pth'), strict=False))
                model.to(device)
            except Exception as e:
                print(f'[Error] Model Loaded Error: {e}')
                exit()

            try:
                with open(f'{strInnerDir}/threshold.json') as f:
                    data = json.load(f)
                    threshold = data['threshold']
                    print(f'Loading Threshold: {threshold}')
            except Exception as e:
                print(f'[Error] Json File Loaded Error: {e}')
                exit()

            model.eval()
            miniPredictions = np.array([])
            miniLabels = np.array([])
            testLoader = DataLoader(dataset = testDataset, batch_size = args.batch_size, shuffle = False, num_workers = args.number_of_workers, pin_memory = False)
            pbar = tqdm(enumerate(testLoader), colour = 'blue', total = len(testLoader))
            for _, data in pbar:
                images, labels, races, _, folders =  data
#                 print(images.shape)
#                 print(labels.shape)
#                 print(races.shape)
                images = images.to(device)
                
                with torch.no_grad():
                    predictions = model(images)
                    intPredictions = torch.ge((predictions.sigmoid().view(-1).detach().to("cpu")), threshold).int()
                    miniPredictions = np.append(miniPredictions, predictions.clone().detach().cpu().numpy())
                    miniLabels = np.append(miniLabels, labels.detach().to("cpu").numpy())
                    for intpred, label, race, folder in zip(intPredictions, labels, races, folders):
                        if folder not in gtResults:
                            gtResults[folder] = label.item()
                        if folder not in predResults:
                            predResults[folder] = [0, 0]
                        if folder not in sensitiveResults:
                            sensitiveResults[folder] = race
                        predResults[folder][intpred] += 1

            _, _, auc, threshold = Find_Optimal_Cutoff(miniLabels.tolist(), miniPredictions.tolist())
            with open(os.path.join(args.model_save_directory, 'testAUCRecode.json'), 'a') as outfile:
                outfile.write(f'{strInnerDir}/weight.pth: {auc}\n')

    json_object = json.dumps(predResults)
    with open(os.path.join(args.model_save_directory, 'predResults.json'), "w") as outfile:
        outfile.write(json_object)

    json_object = json.dumps(gtResults)
    with open(os.path.join(args.model_save_directory, 'groundTruthResults.json'), "w") as outfile:
        outfile.write(json_object)

    multiVote = {}
    for folderName in predResults:
        multiVote[folderName] = predResults[folderName].index(max(predResults[folderName]))

    print('Confusion Matrix Phase')
    predictions = []
    labels = []
    sensitives = []
    for folderName in multiVote:
        predictions.append(int(multiVote[folderName]))
        labels.append(int(gtResults[folderName]))
        sensitives.append(int(sensitiveResults[folderName]))
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    sensitives = torch.tensor(sensitives)

    ##　For all confusionMatrix
    print(f'ACC({torch.eq(labels, predictions).sum()}/{len(labels)}):{torch.eq(labels, predictions).sum()/len(labels)}')
    confusionMatrix = confusion_matrix(labels.tolist(), predictions.tolist())
    _, _ = plot_confusion_matrix(conf_mat = confusionMatrix)
    plt.savefig(os.path.join(args.model_save_directory, 'ALL.png'))

    ##　For Majority confusionMatrix
    MajorityLabels = labels[sensitives == 0]
    MajorityPredictions = predictions[sensitives == 0]
    MajorityACC = torch.eq(MajorityLabels, MajorityPredictions).sum()/len(MajorityLabels)
    
    print(f'Majority ACC({torch.eq(MajorityLabels, MajorityPredictions).sum()}/{len(MajorityLabels)}):{MajorityACC}')
    confusionMatrix = confusion_matrix(MajorityLabels.tolist(), MajorityPredictions.tolist())
    _, _ = plot_confusion_matrix(conf_mat = confusionMatrix)
    plt.savefig(os.path.join(args.model_save_directory, 'Majority.png'))

    ##　For Minority confusionMatrix
    MinorityLabels = labels[sensitives == 1]
    MinorityPredictions = predictions[sensitives == 1]
    MinorityACC = torch.eq(MinorityLabels, MinorityPredictions).sum()/len(MinorityLabels)

    print(f'Minority ACC({torch.eq(MinorityLabels, MinorityPredictions).sum()}/{len(MinorityLabels)}):{MinorityACC}')
    confusionMatrix = confusion_matrix(MinorityLabels.tolist(), MinorityPredictions.tolist())
    _, _ = plot_confusion_matrix(conf_mat = confusionMatrix)
    plt.savefig(os.path.join(args.model_save_directory, 'Minority.png'))

    fairResult = FairnessMetrics(predictions.numpy(), labels.numpy(), sensitives.numpy())

    for i in fairResult:
        print(f'{i}: {fairResult[i]}')
    json_object = json.dumps(fairResult)
    with open(os.path.join(args.model_save_directory, 'fairnessMetrics.json'), "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    try:

        args = parser_setting()
        print(args)

        ### create directory
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

        random.seed(24)
        argsdic = vars(args)
        json_object = json.dumps(argsdic)
        with open(os.path.join(args.model_save_directory, 'args.json'), "w") as outfile:
            outfile.write(json_object)
        
        ### torch device
        if torch.cuda.is_available() and not args.cpu: # 若想使用 cuda 且可以使用 cuda
            device = 'cuda'
        else:
            device = 'cpu'
        print(f'Training on {device}')

        ListDataset = []
        for datasetPath in args.datasetpath:
            ListDataset.append(pd.read_pickle(datasetPath))
        pklPatchesInformation = pd.concat([pd.read_pickle(patchesinformation) for patchesinformation in args.patchesinformation], ignore_index = True)
        
        mainTrainValid(ListDataset, pklPatchesInformation, device, strWanbdGroupName = strRandomDir)
        mainTest(ListDataset, pklPatchesInformation, device)

    except Exception as e:
        print(f'[Error] Initialize the experiment: {e}.')
