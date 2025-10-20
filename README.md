# Contrastive Learning Enhances Fairness in Pathology Artificial Intelligence Systems

Shih-Yen Lin<sup>†</sup>, Pei-Chen Tsai<sup>†</sup>, Fang-Yi Su<sup>†</sup>, Chun-Yen Chen, Fuchen Li, Junhan Zhao, Yuk Yeung Ho, Tsung-Lu Michael Lee, Elizabeth Healey, Po-Jen Lin, Ting-Wan Kao, Dmytro Vremenko, Thomas Roetzer-Pejrimovsky, Lynette Sholl, Deborah Dillon, Nancy U. Lin, David Meredith, Keith L. Ligon, Ying-Chun Lo, Nipon Chaisuriya, David J. Cook, Adelheid Woehrer, Jeffrey Meyerhardt, Shuji Ogino, MacLean P. Nasrallah, Jeffrey A. Golden, Sabina Signoretti, Jung-Hsien Chiang, Kun-Hsing Yu



## Introduction
*AI-enhanced pathology evaluation systems hold significant promise for improving cancer diagnosis but frequently exhibit biases against underrepresented populations due to limited diversity in training data. Here we present the Fairness-aware Artificial Intelligence Review for Pathology (FAIR-Path), a novel framework that leverages contrastive learning and weakly supervised machine learning to mitigate bias in AI-based pathology evaluation. In a pan-cancer AI fairness analysis spanning 20 cancer types, we found significant performance disparities in 29.3% of diagnostic tasks across demographic groups defined by self-reported race, sex, and age. FAIR-Path effectively mitigated 88.5% of these disparities, with external validation showed 91.1% reduction in performance gaps across 15 independent cohorts. We found that variations in somatic mutation prevalence among populations contributed to these performance disparities. FAIR-Path represents a promising step toward addressing fairness challenges in AI-powered pathology diagnoses and provides a robust framework for mitigating bias in medical AI applications.*

![image](https://i.ibb.co/gL70h1Tp/2025-10-07-201353.jpg)

You can find the resource from google drive : https://drive.google.com/drive/u/1/folders/12og_0dCEj6ZJTvQ3oqhFJSRJ-GcbjpuE

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
* Stage 1: Representation Learning
The first stage involves learning patch-level representations.
    * `--datasetpath: Paths to the dataset files for each sensitive attribute and class combination.`
    * `--patchesdirectory: Paths to directories containing image patches.`
    * `--epoch: Number of epochs for training.`
    * `--wandb: Log to WandB for experiment tracking.`
    * `--learning_rate: Set the learning rate for training.`

    
```
python representation_learning.py 
    --datasetpath 'sensitive_attribute0 Class0.pkl' \
	          'sensitive_attribute0 Class1.pkl' \
	          'sensitive_attribute1 Class0.pkl' \
	          'sensitive_attribute1 Class1.pkl' \
    --patchesdirectory 'path to img folder (sensitive_attribute0 Class0)' \
			'path to img folder (sensitive_attribute0 Class1)' \
			'path to img folder (sensitive_attribute1 Class0)' \
			'path to img folder (sensitive_attribute1 Class1)' \
	--patchesinformation 'path img_information.pkl class0' \
                            'path img_information.pkl class1'
	--model_save_directory 'save weight path' \
	--epoch 200 --batch_size 12 --step 480 
    --wandb --wandb_projectname project_name 
    --pickType k-step --multiply 24 --specificInnerloop 2 
    --learning_rate 5e-3
```

    
* Stage 2: Fine-tuning
After the first stage, use the saved model weights to fine-tune for the classification task
    * `--pretraineddirectory: Path to the pretrained model weights from Stage 1.`
```
python mainFinetuneClassificationTask.py 
    --datasetpath 'sensitive_attribute0 Class0.pkl' \
	          'sensitive_attribute0 Class1.pkl' \
	          'sensitive_attribute1 Class0.pkl' \
	          'sensitive_attribute1 Class1.pkl' \
    --patchesdirectory 'path to img folder (sensitive_attribute0 Class0)' \
			'path to img folder (sensitive_attribute0 Class1)' \
			'path to img folder (sensitive_attribute1 Class0)' \
			'path to img folder (sensitive_attribute1 Class1)' \
	--patchesinformation 'path img_information.pkl' \
                            'path img_information.pkl'
	--model_save_directory 'save weight path' \
    --pretraineddirectory 'path to weight from stage 1' \
    --epoch 50 --batch_size 4 
    --wandb --wandb_projectname project name 
    --pickType k-step --specificInnerloop 1 --multiply 3 
    --learning_rate 5e-5
   
```  


## Tutorial
### example for gender in tumor detection 33_CHOL (frozen section)
* Stage 1
    
```
python representation_learning.py 
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
* Stage 2
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
