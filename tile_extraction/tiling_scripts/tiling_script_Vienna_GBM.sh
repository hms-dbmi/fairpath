#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 2:00:00                         # Runtime in D-HH:MM format
#SBATCH -p short                            # Partition to run in
#SBATCH --mem=1G                          # Memory total in MiB (for all cores)
#SBATCH -o ./logs/tile_DFCI_brain/WSI_tile%A_%a.log                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./logs/tile_DFCI_brain/WSI_tile%A_%a.log                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --array=0-474          # Run array for indexes. Should set to the length of the file list

source deactivate
module restore default
source activate SY_clam


## csv file to the WSI file list
## REMEMBER TO MODIFY THE --array TO THE LENGTH OF THE  FILE LIST !!
FILE_LIST="data_sheet/Vienna_GBM(N=474).csv"

IDX=$((SLURM_ARRAY_TASK_ID))  ## file index to run (from job array ID.
OUTPATH="/n/data2/hms/dbmi/kyu/lab/shl968/tile_datasets"    ## main output folder for extracted tiles
TILE_PARAMS="step1_tile_extraction/tiling_params/tile_params_quick500_w512s512_satOtsu.jsonc"        ## tile param file
PROJ="Vienna_GBM_PM"                ## output subfolder name. the tiles will be stored in [outpath]/[proj]/[slide_name]

python step1_tile_extraction/WSI_tile_extraction_batch.py \
 --params ${TILE_PARAMS} \
 --file_list ${FILE_LIST} --file_index ${IDX} \
 --outpath ${OUTPATH}  --proj ${PROJ} 

