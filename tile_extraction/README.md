# Tile Extraction

**Dependencies**

* Install [HistoLab](https://github.com/histolab/histolab) for cellularity estimation.

## Step 1: Create a list of the whole slide images (WSIs) you would like to tile

* (see `generate_tiling_csv.ipynb` for example)

## Step 2: Perform tiling using the WSI list

Be sure to specify the following arguments:
 * `--file_list`: The WSI list mentioned above
 * `--params`: The tiling parameter file (see `tiling_params/tile_params_w512s512.jsonc` for example)
 * `--outpath`: Output directory
 * `--proj`: dataset name (also will be the subfolder name)
 
Excute the following python code:
```
python WSI_tile_extraction_batch.py \
 --params [TILE_PARAMS] \
 --file_list [FILE_LIST_CSV] \
 --outpath [OUTPATH]  --proj [PROJ]
```

Alternatively, if you are working under a Slurm Workload Manager, you could submit the following slurm scripts:

```
sbatch step1_tile_extraction/tiling_scripts/tiling_script_CPTAC_LUAD.sh
```

The output will be stored in `[OUTPATH]/[PROJ]`, which will contain 3 subfolders :

* `thumbnail`: thumbnails for WSIs
* `overlay_vis`: thumbnails with all valid tiles highlighted. Can be used for troubleshooting.
* `overlay_vis_top[K]`: thumbnails with the selected top [K] tiles highlighted. Can be used for troubleshooting.
* `tileStats`: Spreadsheets storing the coordinates of the extracted tiles (which will be used in the next steps). Each WSI will have its own .csv file.

