#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 2:00:00                         # Runtime in D-HH:MM format
#SBATCH -p short                            # Partition to run in
#SBATCH --mem=1G                          # Memory total in MiB (for all cores)
#SBATCH -o ./logs/tile_PLCO_colon/WSI_tile%A_%a.log                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./logs/tile_PLCO_colon/WSI_tile%A_%a.log                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --array=0-332          # Run array for indexes. Should set to the length of the file list


# module load conda3/latest   cuda/10.2   libpng/1.6.26 tiff/4.0.7 \
#     libxml/2.9.4    libffi/3.2.1 openjpeg/2.2.0 \
#     gcc/6.2.0   python/3.6.0    jpeg/9b     glib/2.50.2     freetype/2.7 \
#     fontconfig/2.12.1  harfbuzz/1.3.4 cairo/1.14.6  openslide/3.4.1

source activate HTAN_env
## csv file to the WSI file list
## REMEMBER TO MODIFY THE --array TO THE LENGTH OF THE  FILE LIST !!
FILE_LIST="data_sheet/PLCO_colon(N=333).csv"

IDX=$((SLURM_ARRAY_TASK_ID))  ## file index to run (from job array ID.
OUTPATH="/n/data2/hms/dbmi/kyu/lab/shl968/tile_datasets"    ## main output folder for extracted tiles
TILE_PARAMS="step1_tile_extraction/tiling_params/tile_params_quick500_w256s128.jsonc"        ## tile param file
PROJ="PLCO_colon"                ## output subfolder name. the tiles will be stored in [outpath]/[proj]/[slide_name]

python step1_tile_extraction/WSI_tile_extraction_batch.py \
 --params ${TILE_PARAMS} \
 --file_list ${FILE_LIST} --file_index ${IDX} \
 --outpath ${OUTPATH}  --proj ${PROJ} --overwrite_if_exist

