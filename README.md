# Contrastive Learning Enhances Fairness in Pathology Artificial Intelligence Systems

Shih-Yen Lin`<sup>`†`</sup>`, Pei-Chen Tsai`<sup>`†`</sup>`, Fang-Yi Su`<sup>`†`</sup>`, Chun-Yen Chen, Fuchen Li, Junhan Zhao, Yuk Yeung Ho, Tsung-Lu Michael Lee, Elizabeth Healey, Po-Jen Lin, Ting-Wan Kao, Dmytro Vremenko, Thomas Roetzer-Pejrimovsky, Lynette Sholl, Deborah Dillon, Nancy U. Lin, David Meredith, Keith L. Ligon, Ying-Chun Lo, Nipon Chaisuriya, David J. Cook, Adelheid Woehrer, Jeffrey Meyerhardt, Shuji Ogino, MacLean P. Nasrallah, Jeffrey A. Golden, Sabina Signoretti, Jung-Hsien Chiang, Kun-Hsing Yu

You can find the resource from google drive : https://drive.google.com/drive/u/1/folders/12og_0dCEj6ZJTvQ3oqhFJSRJ-GcbjpuE

![image](https://i.ibb.co/gL70h1Tp/2025-10-07-201353.jpg)

## Requirements

    * docker version: nvcr.io/nvidia/pytorch:22.03-py3
    * albumentations==1.3.0
    * mlxtend==0.20.0
    * numpy==1.22.4
    * opencv-python-headless==4.5.5.64
    * pandas==1.4.4
    * scikit-learn==1.1.2
    * wandb==0.13.2
    * warmup_scheduler

## Whole Slide Images Tiling

* 20x
* 512*512

## Data Preparation

* datasetpath

  * fairpath real data pkl file : https://drive.google.com/drive/u/1/folders/1VGlj06b9UQgdVxcakhoTvYhy0zitQF1g
    ![image](https://i.imgur.com/hMXp7HQ.png)
* patchesdirectory

  * fairpath example of data format : https://drive.google.com/drive/u/1/folders/1yFTQ8Vc9VqFXRy-oCDrBgafeYDoSQ-3m
    ![image](https://i.imgur.com/Qe9DGsU.png)
* patchesinformation

  * fairpath real patches information : https://drive.google.com/drive/u/1/folders/1XQF2zcTr5zwMvWza9w1C0rC_V_FEY1NY
    ![image](https://i.imgur.com/SW13jlE.png)

## Usage

* Stage 1 (Pre-training) **[XXX TO BE UPDATED XXXX]**

```
python mainRepresentationLearningNonnorm.py 
    --datasetpath 'sensitive_attibute0 Class0.pkl' \
	          'sensitive_attibute0 Class1.pkl' \
	          'sensitive_attibute1 Class0.pkl' \
	          'sensitive_attibute1 Class1.pkl' \
    --patchesdirectory 'path to img folder (sensitive_attibute0 Class0)' \
			'path to img folder (sensitive_attibute0 Class1)' \
			'path to img folder (sensitive_attibute1 Class0)' \
			'path to img folder (sensitive_attibute1 Class1)' \
	--patchesinformation 'path img_information.pkl class0' \
                            'path img_information.pkl class1'
	--model_save_directory 'save weight path' \
	--epoch 200 --batch_size 12 --step 480 
    --wandb --wandb_projectname project_name 
    --pickType k-step --multiply 24 --specificInnerloop 2 
    --learning_rate 5e-3
```

* Stage 2 (feature extraction) **[XXX TO BE UPDATED XXXX]**

```
python mainFinetuneClassificationTask.py 
    --datasetpath 'sensitive_attibute0 Class0.pkl' \
	          'sensitive_attibute0 Class1.pkl' \
	          'sensitive_attibute1 Class0.pkl' \
	          'sensitive_attibute1 Class1.pkl' \
    --patchesdirectory 'path to img folder (sensitive_attibute0 Class0)' \
			'path to img folder (sensitive_attibute0 Class1)' \
			'path to img folder (sensitive_attibute1 Class0)' \
			'path to img folder (sensitive_attibute1 Class1)' \
	--patchesinformation 'path img_information.pkl' \
                            'path img_information.pkl'
	--model_save_directory 'save weight path' \
    --pretraineddirectory 'path to weight from stage 1' \
    --epoch 50 --batch_size 4 
    --wandb --wandb_projectname project name 
    --pickType k-step --specificInnerloop 1 --multiply 3 
    --learning_rate 5e-5
   
```

* **Stage 3 (Fine-Tuning using corrected feature)**

```
python mainFinetuneClassificationTask.py --folder [FOLDER_FOR_STORING_FEATURES]
```


## Tutorial

### example for gender in tumor detection 33_CHOL (frozen section)

* Stage 1 (Pre-training) **[XXX TO BE UPDATED XXXX]**

```
python mainRepresentationLearningNonnorm.py 
    --datasetpath 'female 33_CHOL 40 Frozen tumor0.pkl' \
	          'female 33_CHOL 40 Frozen tumor1.pkl' \
	          'male 33_CHOL 40 Frozen tumor0.pkl' \
	          'male 33_CHOL 40 Frozen tumor1.pkl' \
    --patchesdirectory '33_CHOL' \
			'33_CHOL' \
			'33_CHOL' \
			'33_CHOL' \
	--patchesinformation 'img_information_20x.pkl' \
                            'img_information_20x.pkl'
	--model_save_directory 'save weight path' \
	--epoch 200 --batch_size 12 --step 480 
    --wandb --wandb_projectname project_name 
    --pickType k-step --multiply 24 --specificInnerloop 2 
    --learning_rate 5e-3
```

* **Stage 2 (feature extraction) **[XXX TO BE UPDATED XXX**]**

```
python mainFinetuneClassificationTask.py 
    --datasetpath 'female 33_CHOL 40 Frozen tumor0.pkl' \
	          'female 33_CHOL 40 Frozen tumor1.pkl' \
	          'male 33_CHOL 40 Frozen tumor0.pkl' \
	          'male 33_CHOL 40 Frozen tumor1.pkl' \
    --patchesdirectory '33_CHOL' \
			'33_CHOL' \
			'33_CHOL' \
			'33_CHOL' \
	--patchesinformation 'img_information_20x.pkl' \
                            'img_information_20x.pkl'
	--model_save_directory 'save weight path' \
    --pretraineddirectory 'path to weight from stage 1' \
    --epoch 50 --batch_size 4 
    --wandb --wandb_projectname project name 
    --pickType k-step --specificInnerloop 1 --multiply 3 
    --learning_rate 5e-5
   
```

* **Stage 3 (Fine-Tuning using corrected feature)**

```
python mainFinetuneClassificationTask.py --folder [FOLDER_FOR_STORING_FEATURES]
```
