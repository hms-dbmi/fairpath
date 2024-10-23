# FairPath
Contrastive Learning Enhances Fairness in Pathology Artificial Intelligence Systems: A Pan-Cancer Multi-Institutional Study


![image](https://ibb.co/tJ5h02f)

![](https://i.imgur.com/qm4OLtI.png)

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
![](https://hackmd.io/_uploads/HJZdLyIlkx.png)

* patchesdirectory
![](https://hackmd.io/_uploads/SJKqa0rlye.png)

* patchesinformation
![](https://hackmd.io/_uploads/Sk8RS1IxJg.png)






## Usage
* Stage 1
    
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
	--patchesinformation 'path img_information.pkl' \
                            'path img_information.pkl'
	--model_save_directory 'save weight path' \
	--epoch 200 --batch_size 12 --step 480 
    --wandb --wandb_projectname project_name 
    --pickType k-step --multiply 24 --specificInnerloop 2 
    --learning_rate 5e-3
```
* Stage 2
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
