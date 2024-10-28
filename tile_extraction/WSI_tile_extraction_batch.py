## WSI_tile_extraction.py

import matplotlib.pyplot as plt
import os
import openslide
import numpy as np
import math
from jsonc_parser.parser import JsoncParser
from PIL import Image
from tqdm import tqdm
from skimage.feature import blob_log, blob_dog, blob_doh
from scipy import ndimage
from skimage.color import rgb2gray
from argparse import ArgumentParser, Namespace
import pandas as pd
from WSI_tile_extraction import tile_WSI
 



if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments

    parser.add_argument(
        "--file_list",
        help="""csv file with the list of WSI files """,
        type=str,
        default= "/n/data2/hms/dbmi/kyu/lab/shl968/fairness_external_validation/data_sheet/DFCI_breast(N=1255).csv" ,
    )

    parser.add_argument(
        "--file_index",
        help="""index of the file to run (from the file_list csv file). If not specified, run all files in the list.""",
        type=int,
        nargs='*',
        # default=None,
        default=np.arange(0,40).tolist(),
    )

    parser.add_argument(
        "--params",
        help="""Param File """,
        type=str,
        default="/n/data2/hms/dbmi/kyu/lab/shl968/fairness_external_validation/step1_tile_extraction/tiling_params/tile_params_w512s512.jsonc",
    )

    parser.add_argument(
        "--outpath",
        help="""Output Path """,
        type=str,
        default="/n/data2/hms/dbmi/kyu/lab/shl968/tile_datasets/",
    )
    parser.add_argument(
        "--proj", help="""Project name (folder prefix) """, type=str, default="Debug"
    )
    parser.add_argument(
        "--blob_fun", help="""Blob function  """, type=str, default="log",choices=["log",'dog','doh']
    )
    parser.add_argument(
        "--overwrite_if_exist",
        help="""Overwrite file is already existed """,
        action='store_true',
        default=True,
    )
    parser.add_argument(
        "--output_tiles",
        help="""Whether to Output tiles (Tiles may use up a lot of storage. You can always find the tiles by the coordinate csv stored in tileStats folders)""",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--read_tile_csv",
        help="""Read existing tilestat spreadsheets instead of searching for the tile from scratch.""",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--debug", help="""Debug using small dataset""", type=bool, default=False
    )
    args = parser.parse_args()

    print("=========================")
    print("    Input Parameters:    ")
    for key,val in vars(args).items():
        print(f'{key}:\t{val}')

    df = pd.read_csv(args.file_list,header=None)
    file_list = list(df[0])
    params = args
    if args.file_index is not None:
        if isinstance( args.file_index, list ):
            for file_index in args.file_index:
                assert file_index < len(file_list), f"The file index ({file_index}) exceeds the length of the file list ({len(file_list)})"
                params = args
                params.infile = file_list[file_index]
                tile_WSI(params)
        elif isinstance( args.file_index, int ):
            file_index = args.file_index
            assert file_index < len(file_list), f"The file index ({file_index}) exceeds the length of the file list ({len(file_list)})"
            params = args
            params.infile = file_list[file_index]
            tile_WSI(params)
        else:
            raise ValueError("file_index should be either an integer or a list of integers")
    else:
        print("file_index not specified. Will run all slides in the list")
        for file_index in range(len(file_list)):
            params = args
            params.infile = file_list[file_index]
            tile_WSI(params)

