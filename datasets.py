## system library
import os

## data library
import numpy as np
import pandas as pd

## torch, sklearn library
import torch
from torch.utils.data import Dataset

## images
import cv2
import albumentations as albu

import random

import h5py
from PIL import Image
from io import BytesIO


class PathologyDataset(Dataset):
    def __init__(self, boxes, patchesDirectory, pklPatchesInformation, patchesPerBag = 200, size = 224, padding = 256, multiply = 24, transform = None, secondtrasform = None, pickType = 'test', mergeDataset = True, step = 100, noise = False, roi = False):
        '''
        Parameters:
            boxes: number of dataset
            patchesDirectory:
            pklPatchesInformation: patch information pickle
            patchesPerBag: Literally
            size: Literally
            padding: image size padding and centeral cut to size
            multiply: sample patchesPerBag*multiply
            transform: Literally
            secondtrasform: Literally
            pickType: normal, k-step, random
            step: __len__
            noise: sample white patches
            roi: if we have roi file, you can choose it.
        '''
        if len(boxes) != len(patchesDirectory):
            raise ValueError(f"The length of boxes list({len(boxes)} is not as same as patchesDirectory list---{len(patchesDirectory)}---)")

        self.size = size
        self.samples = []
        self.samplesPath = []
        self.targets = []
        self.races = []
        self.genders = []
        self.folderDict = {}


        if mergeDataset == True:
            for boxidx, samples in enumerate(boxes):


                self.samples += [s[3] for s in samples]
#                 self.samplesPath += [f'{patchesDirectory[boxidx]}{s[3]}' for s in samples]
                # example for each item: 04_LUAD/TCGA-67-3773-01Z-00-DX1.3E9DFC22-E962-4C18-BF5B-27EBEA089F5D
                self.samplesPath += ([f'{patchesDirectory[boxidx]}{s[3]}'[12:] for s in samples])
                # TODO use this targets to find the pos_weight
                self.targets += [s[0] for s in samples]
                self.races += [s[1] for s in samples]
                self.genders += [s[2] for s in samples]
            self.step = len(self.samples)
        else:
            for boxidx, samples in enumerate(boxes):

                self.samples.append([s[3] for s in samples])
#                 self.samplesPath.append([f'{patchesDirectory[boxidx]}{s[3]}' for s in samples])


                self.samplesPath.append([f'{patchesDirectory[boxidx]}{s[3]}'[12:] for s in samples])

                self.targets.append([s[0] for s in samples])
                self.races.append([s[1] for s in samples])
                self.genders.append([s[2] for s in samples])
            self.step = step

        self.singleDataset = mergeDataset

        self.pklPatchesInformation = pklPatchesInformation
        self.patchesPerBag = patchesPerBag

        self.transform = transform
        if secondtrasform == None:
            self.secondtrasform = transform
        else:
            self.secondtrasform = secondtrasform
        self.image_padding = albu.Compose([
            albu.PadIfNeeded(min_height = padding, min_width = padding, border_mode = cv2.BORDER_CONSTANT, value = (0, 0, 0), p = 1),
            albu.CenterCrop(height = size, width = size, p = 1)
        ], p = 1)


        self.pickType = pickType.lower()
        self.multiply = multiply
        self.noise = noise
        self.roi = roi
        if self.pickType == 'test':
            self.multiply = 1

    def __len__(self):
        return self.step

    def openImg(self, path, transform = None):
#         print(self.hf)
#         if os.path.isfile(path) is True:#原路徑../TCGA_IMG/01_BRCA/...    把前面拿掉才能對照hf 故改Fslae
        if len(path[:path.find('/')]) !=0:
#             print(path)

            hf = h5py.File('/n/data2/hms/dbmi/kyu/lab/datasets/TCGA_sophie/'+path[:path.find('/')]+'.hdf5', 'r')
#             print('../TCGA_IMG/'+path[:path.find('/')]+'.hdf5')


            kk = np.array(hf[path])
            img = Image.open(BytesIO(kk))
            img = np.array(img)

#             img = cv2.imread(img, 1)



            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f'[Error] {path} is {type(img)}.')
                exit()
            if transform == None:
                img = self.image_padding(image = img)["image"]
            else:
                img = transform(image = img)["image"]
            img = np.moveaxis(img, -1, 0)
        else:
#             hf = h5py.File('../TCGA_IMG/'+path[:path.find('/')]+'.hdf5', 'r')
#             kk = np.array(self.hf[path])
#             img = Image.open(BytesIO(kk))
#             img = np. array(img)

            if path != '':
                raise ValueError(f'[Error] patches loaded error: {path}')
            img = np.zeros([3, self.size, self.size])
        return img

    def pickPatches(self, sampleDirectory, folder):
        '''
        return a list of patches' path for a given sample (folder)
        '''
        self.imgsize=512
#         print(self.folderDict)
        if folder not in self.folderDict:
            # sample pathches at first time
            # sample size: patchesPerBag*multiply. see Line 122

#             if self.roi == True:

#                 listPatches = self.pklPatchesInformation.loc[self.pklPatchesInformation["folder_id"].isin([folder])].sort_values(by=["foreground_background_percentage"], ascending = False).head(self.patchesPerBag*self.multiply)["figure_id"].tolist()
#             else:
#                 if self.noise == False:
#                     listPatches = self.pklPatchesInformation.loc[self.pklPatchesInformation["folder_id"].isin([folder]) ].sort_values(by=["foreground_background_percentage"], ascending = False).head(self.patchesPerBag*self.multiply)["figure_id"].tolist()
#                 else:
#                     listPatches = self.pklPatchesInformation.loc[self.pklPatchesInformation["folder_id"].isin([folder])].sort_values(by=["foreground_background_percentage"], ascending = False).head(self.patchesPerBag*self.multiply)["figure_id"].tolist()

#             self.folderDict[folder] = listPatches
#         else:
#             listPatches = self.folderDict[folder]



            if self.roi == True:

                listPatches = self.pklPatchesInformation.loc[self.pklPatchesInformation["folder_id"].isin([folder]) & self.pklPatchesInformation['height'].isin([self.imgsize]) & self.pklPatchesInformation['width'].isin([self.imgsize])].sort_values(by=["weight"], ascending = False).head(self.patchesPerBag*self.multiply)["figure_id"].tolist()
            else:
                if self.noise == False:
                    listPatches = self.pklPatchesInformation.loc[self.pklPatchesInformation["folder_id"].isin([folder]) & self.pklPatchesInformation['height'].isin([self.imgsize]) & self.pklPatchesInformation['width'].isin([self.imgsize])].sort_values(by=["foreground_background_percentage"], ascending = False).head(self.patchesPerBag*self.multiply)["figure_id"].tolist()
                else:
                    listPatches = self.pklPatchesInformation.loc[self.pklPatchesInformation["folder_id"].isin([folder])].sort_values(by=["foreground_background_percentage"], ascending = False).head(self.patchesPerBag*self.multiply)["figure_id"].tolist()

            self.folderDict[folder] = listPatches
        else:
            listPatches = self.folderDict[folder]



        # sample strategy:
            # test: multiply=1
            # k-step:
                # assume patchesPerBag(100)*multiply(5) = 500
                # section = 5, and random->3, listPatches = idx([3, 8, ..., 498]) (498-3)/5+1 = 100,
                #                                                                   [:self.patchesPerBag] for more than 100.
        if self.pickType == 'test':
            pass
            ### because test mode only choose self.patchesPerBag of images
        elif self.pickType == 'k-step':
            if self.patchesPerBag < len(listPatches):
                section = len(listPatches)//self.patchesPerBag
                listPatches = listPatches[random.randint(0, section-1)::section][:self.patchesPerBag]
        elif self.pickType == 'random':
            listPatches = random.sample(listPatches, min(self.patchesPerBag, len(listPatches)))
        else:
            raise ValueError(f"The {self.pickType} method to pick pathches isn't implementation yet. :( ")

        # Add path.
        ListPatchesPath = [os.path.join(sampleDirectory, patch) for patch in listPatches]
#         print(ListPatchesPath)
        if len(ListPatchesPath) == 0:
            raise ValueError(f'This folder is empty. {folder}.')
        if self.patchesPerBag > len(ListPatchesPath):
            ListPatchesPath += ['']*(self.patchesPerBag - len(ListPatchesPath))

        return ListPatchesPath

    def __getitem__(self, idx):



        # singleDataset
        if self.singleDataset:
            folder = self.samples[idx]
            sampleDirectory = self.samplesPath[idx]
            label  = self.targets[idx]
            race  = self.races[idx]
            gender = self.genders[idx]

            ListPatchesPath = self.pickPatches(sampleDirectory, folder)
            # containing all required patches of one sample
            imgs = np.asarray([self.openImg(i, self.transform) for i in ListPatchesPath])

            return torch.Tensor(imgs)/255.0, torch.tensor(int(label)), torch.tensor(int(race)), torch.tensor(int(gender)), folder
        else:
        # sample one in every dataset.
            labels  = []
            folders = []
            races   = []
            genders = []
            imgs = np.empty((0, 3, self.size, self.size))
            for idxFolder in range(len(self.samples)):
                randomNum = random.randint(0, len(self.samples[idxFolder])-1)
                folder = self.samples[idxFolder][randomNum]
                sampleDirectory = self.samplesPath[idxFolder][randomNum]
                label  = self.targets[idxFolder][randomNum]
                race   = self.races[idxFolder][randomNum]
                gender = self.genders[idxFolder][randomNum]
                ListPatchesPath = self.pickPatches(sampleDirectory, folder)

                img      = np.asarray([self.openImg(i, self.transform)      for i in ListPatchesPath])
                imgsprime = np.asarray([self.openImg(i, self.secondtrasform) for i in ListPatchesPath])
                imgs = np.append(imgs, img, axis = 0)
                imgs = np.append(imgs, imgsprime, axis = 0)
                labels.append(int(label))
                races.append(int(race))
                genders.append(int(gender))
                folders.append(folder)
            strFolders = '/'.join(folders)
            
            
            return (torch.Tensor(imgs)/255.0).reshape(-1, self.patchesPerBag, 3, self.size, self.size), torch.tensor(labels), torch.tensor(races), torch.tensor(genders), strFolders
