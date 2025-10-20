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
# import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

## warmup scheduler
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

## wandb
import wandb

## written library
from networks import RepNet, MultiRepNet
from datasets import PathologyDataset
from loss import ContrastiveLoss
import util
from framework import parser_setting, create_directory, dataset_generator

## random
import string
import random
from datetime import datetime

## argument library
import argparse

PatchesPerBag = 10

def run(dataloader, epo, epoch, model, criterion = None, optimizer = None,  device = 'gpu'):
    if model.training == True: ## Here for training
        color = 'green'
    else: ## Here for validation
        color = 'red'
    lenSamples = 0
    totalLoss = 0
    totalcateLoss = 0
    totalraceLoss = 0

    pbar = tqdm(enumerate(dataloader), colour = color, total = len(dataloader))
    for _, data in pbar:
        images, labels, races, _, _ =  data
        bz, sz, _, _, _, _ = images.shape # batch size, sample size, patches size, ...
        images = images.reshape(-1, *images.shape[2:]).to(device) ## [bz*sz, pz, c, h, w]
        
        ##
        if model.training == True:
            optimizer.zero_grad()
            # features, secondfeatures = model(images) ## [bz*sz, 128] (2*(6*2), 128) -> 
            features = model(images)
        else:
            with torch.no_grad():
                features = model(images) ## [bz*sz, 128]
        ## norm features
        normalized_features = nn.functional.normalize(features, dim = 1)
        # div_features = torch.div(features, torch.sqrt(torch.tensor(features.shape[-1])))
        div_features = features.clone()
        normalized_features = normalized_features.reshape(bz*sz//2, 2, -1)
        div_features = div_features.reshape(bz*sz//2, 2, -1)

        # secondfeatures = features.reshape(bz*sz//2, 2, -1)

        labels = torch.flatten(labels)
        races = torch.flatten(races)
        
        ## calculate loss
        
        lossCate = criterion(normalized_features, labels, method = 'SupCon')
        lossRace = criterion(div_features, labels = labels, sensitive = races)
        # lossRace = criterion(secondfeatures, labels = labels, sensitive = races)

        lenSamples += len(features)
        loss = lossCate+lossRace
        # loss = (lossCate+lossRace)
        totalLoss += loss.clone().detach().cpu().numpy()*len(features)
        totalcateLoss += lossCate.clone().detach().cpu().numpy()*len(features)
        totalraceLoss += lossRace.clone().detach().cpu().numpy()*len(features)

        ## calculate grad and update
        if model.training == True:
            loss.backward()
            optimizer.step()       
        
        ## set pbar description
        if model.training == True:
            strTraining = "Train"
        else:
            strTraining = "Valid"
        pbar.set_description(
            f'{strTraining} Iter: {epo+1:03}/{epoch:03}  '
            f'Loss: {loss:3.4f} '
            f'CateLoss: {lossCate:3.4f} '
            f'RaceLoss: {lossRace:3.4f} '
        )
        pbar.update()

    return totalLoss/lenSamples, totalcateLoss/lenSamples, totalraceLoss/lenSamples

def mainTrainValid(ListDataset, pklPatchesInformation, device = 'cuda', strWanbdGroupName = ''):
       
    ### k-fold
    for IdxOuterLoop in range(4):
        for IdxInnerLoop in range(1):
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
            
            ListTrainSet = [samples['fnlist'][IdxOuterLoop][0][IdxInnerLoop][0][:] for samples in ListDataset]
            # ListValidSet = [samples['fnlist'][IdxOuterLoop][0][IdxInnerLoop][1][:] for samples in ListDataset]
            
            trainDataset = PathologyDataset(
                boxes = ListTrainSet,
                patchesDirectory = args.patchesdirectory,
                pklPatchesInformation = pklPatchesInformation,
                patchesPerBag = PatchesPerBag,
                size = 224,
                pickType = args.pickType,
                transform = util.image_transform(224),
                secondtrasform = util.light_image_transform(224),
                mergeDataset = False,
                step = args.step,
                multiply = args.multiply
            )
            # validDataset = PathologyDataset(
            #     boxes = ListValidSet,
            #     patchesDirectory = args.patchesdirectory,
            #     pklPatchesInformation = pklPatchesInformation,
            #     patchesPerBag = PatchesPerBag,
            #     size = 224,
            #     pickType = 'test',
            #     mergeDataset = False,
            #     step = 50
            # )

            trainLoader = DataLoader(dataset = trainDataset, batch_size = args.batch_size, shuffle = True, num_workers = args.number_of_workers, pin_memory = False)
            # validLoader = DataLoader(dataset = validDataset, batch_size = args.batch_size, shuffle = False, num_workers = 16, pin_memory = False)
            
            # model = MultiRepNet(pretrained = True)
            model = RepNet(pretrained = True, projectTytpe = 'mlp', feature_normalized = False)

            if args.pretraineddirectory != '':
                print(model.load_state_dict(torch.load(f'./weight/{args.pretraineddirectory}/{IdxOuterLoop}_{IdxInnerLoop}/weight.pth'), strict=False))
                print(f'Model Loaded ./weight/{args.pretraineddirectory}/{IdxOuterLoop}_{IdxInnerLoop}/weight.pth.')
            torch.save(model.state_dict(), os.path.join(strInnerDir, 'weight.pth'))
            
            model.to(device)
            
            optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9)

            warm_up_epoches = 1
            scheduler_steplr = CosineAnnealingLR(optimizer, args.epoch)
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier = 1, total_epoch = warm_up_epoches, after_scheduler = scheduler_steplr)
            
            criterion = ContrastiveLoss()
            
            ####################### Main Stage #########################
            ## For scheduler ##
            optimizer.zero_grad()
            optimizer.step()
            # BestLoss = 1e12
            for epo in range(args.epoch):
                ### Trainin phase
                model.train()
                scheduler_warmup.step()
                trainLoss, catLoss, raceLoss = run(trainLoader, epo, args.epoch, model, criterion, optimizer, device = device)

                # model.eval()
                # validLoss = run(validLoader, epo, args.epoch, model, criterion, device = device)

                # if BestLoss > validLoss:
                #     BestLoss = validLoss
                #     torch.save(model.state_dict(), os.path.join(strInnerDir, 'weight.pth'))
                #     print('Update Loss model ! ', end = '')
                
                
#                 if (epo+1) >= 200 and (epo+1)%50 == 0:
#                     print(f"Update Epoch: {epo+1}", end = ' ')
#                     torch.save(model.state_dict(), os.path.join(strInnerDir, f'weight{epo+1}.pth'))
                    
                #print(args.epoch)    
                if epo+1==args.epoch:              
                    print(f"Update Epoch: {epo+1}", end = ' ')
                    torch.save(model.state_dict(), os.path.join(strInnerDir, f'weight{epo+1}.pth'))     
                    

                # print(f"Train Loss: {trainLoss} Validation Loss: {validLoss}")
                try:
                    if args.wandb == True:
                        ## method 1
                        wandb.log({
                            "train_loss": trainLoss,
                            "categorical loss": catLoss,
                            "racial loss": raceLoss
                            # "valid_loss": validLoss,
                        })
                except Exception as e:
                    print(f'[Error] Wandb record: {e}')
                    exit()

            if args.wandb == True:
                wandb.finish()

    return strInnerDir




def extract_feature(ListDataset, pklPatchesInformation, device):

    gtResults = {}
    predResults = {}
    sensitiveResults = {}
    for IdxOuterLoop in range(args.outerloop):
        

        for IdxInnerLoop in range(args.innerloop):
            ListTrainSet, ListValidSet = dataset_generator(args, IdxOuterLoop, IdxInnerLoop, ListDataset)
            ListTestSet = [samples['fnlist'][IdxOuterLoop][1][:] for samples in ListDataset]
            AllSet = [a + b + c for a, b, c in zip(ListTrainSet, ListValidSet, ListTestSet)]
            # AllSet = ListTrainSet+ListValidSet+ListTestSet
            
            PatchesPerBag = 200

            testDataset = PathologyDataset(
                boxes = AllSet,
                patchesDirectory = args.patchesdirectory,
                pklPatchesInformation = pklPatchesInformation,
                patchesPerBag = PatchesPerBag,
                size = 224,
                pickType = 'test',
                transform = None,
                mergeDataset = True,
            )            
            if args.specificInnerloop != -1 and args.innerloop == 1:
                IdxInnerLoop = args.specificInnerloop
            try:
                strInnerDir = os.path.join(args.model_save_directory, f'{IdxOuterLoop}_{IdxInnerLoop}')
                model = RepNet(pretrained = True, projectTytpe = 'mlp', feature_normalized = False)
                print(model.load_state_dict(torch.load(f'{args.pretraineddirectory}/{IdxOuterLoop}_{IdxInnerLoop}/weight{args.epoch}.pth'), strict=False))
                print(f'Model Loaded {args.pretraineddirectory}/{IdxOuterLoop}_{IdxInnerLoop}/weight{args.epoch}.pth.')
                # print(f'Loading model: {strInnerDir}')
                # print(model.load_state_dict(torch.load(f'{strInnerDir}/weight.pth'), strict=False))
                model.to(device)
            except Exception as e:
                print(f'[Error] Model Loaded Error: {e}')
                exit()

            # try:
            #     with open(f'{strInnerDir}/threshold.json') as f:
            #         data = json.load(f)
            #         threshold = data['threshold']
            #         print(f'Loading Threshold: {threshold}')
            # except Exception as e:
            #     print(f'[Error] Json File Loaded Error: {e}')
            #     exit()

            model.eval()
            miniPredictions = np.array([])
            probPredictions = np.array([])
            
            miniLabels = np.array([])
            testLoader = DataLoader(dataset = testDataset, batch_size = args.batch_size, shuffle = False, num_workers = args.number_of_workers, pin_memory = False)
            pbar = tqdm(enumerate(testLoader), colour = 'blue', total = len(testLoader))
            for _, data in pbar:
                
                images, labels, races, _, folders =  data
                images = images.to(device)
                with torch.no_grad():
                    predictions = model(images)
                    # predictions = model(images)
                    # intPredictions = torch.ge((predictions.sigmoid().view(-1).detach().to("cpu")), threshold).int()
                    # miniPredictions = np.append(miniPredictions, predictions.clone().detach().cpu().numpy())
                 
                    # probPredictions = np.append(probPredictions, predictions.cpu().numpy())
                    
                    output_dir=args.model_save_directory+'/feature_new'+str(IdxOuterLoop)+'_'+str(IdxInnerLoop)
                    os.makedirs(output_dir, exist_ok=True)

                    for i, folder_name in enumerate(folders):

                        file_path = os.path.join(output_dir, f"{folder_name}.pt")
                        # file_path = os.path.join(args.model_save_directory, 'feature_test'+str(IdxOuterLoop)+'_'+str(IdxInnerLoop)+'/'+folder_name+'.pt')
                    
                        torch.save(predictions[i], file_path)  # Save as .pt file




if __name__ == '__main__':
    try:
        args = parser_setting()
        argsdic = vars(args)
        json_object = json.dumps(argsdic)
        print(args)
        strRandomDir = create_directory(args)
        ## create directory
        # try:
        #     random.seed(datetime.now().timestamp())
        #     if not (os.path.exists(args.model_save_directory) or os.path.isdir(args.model_save_directory)):
        #             os.mkdir(args.model_save_directory)
        #     while 1:
        #         strRandomDir =  random.choices(string.ascii_letters, k=10)
        #         strRandomDir = "".join(strRandomDir)
        #         strSavedWeightDir = f'{args.model_save_directory}/{strRandomDir}'
        #         if not os.path.exists(strSavedWeightDir) or not os.path.isdir(strSavedWeightDir):
        #             os.mkdir(strSavedWeightDir)
        #             print(strSavedWeightDir)
        #             args.model_save_directory = strSavedWeightDir
        #             break
        # except Exception as e:
        #     print(f'[Error] Create the model saved directory: {e}.')
        #     exit()
        ## save args
        with open(os.path.join(args.model_save_directory, 'args.json'), "w") as outfile:
            outfile.write(json_object)
        
        ### torch device
        if torch.cuda.is_available(): # 若想使用 cuda 且可以使用 cuda
            device = 'cuda'
        else:
            device = 'cpu'
        print(f'Training on {device}')

        ListDataset = []
        for datasetPath in args.datasetpath:
            ListDataset.append(pd.read_pickle(datasetPath))
        print("Training Dataset")
        for i in ListDataset:
            print(i)
            print('----------------')
            print(len(i['fnlist'][0][1]))
        pklPatchesInformation = pd.concat([pd.read_pickle(patchesinformation) for patchesinformation in args.patchesinformation], ignore_index = True)
        
        mainTrainValid(ListDataset, pklPatchesInformation, device, strWanbdGroupName = strRandomDir)
        extract_feature(ListDataset, pklPatchesInformation, device)
        

    except Exception as e:
        print(f'[Error] Initialize the experiment: {e}.')
