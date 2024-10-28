# WSI_tile_extraction.py

import matplotlib.pyplot as plt
import os
import openslide
import numpy as np
import math
from jsonc_parser.parser import JsoncParser
from PIL import Image, ImageOps
from tqdm import tqdm
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.feature import blob_log, blob_dog, blob_doh
from scipy import ndimage
from skimage.color import rgb2gray, rgb2hsv
from argparse import ArgumentParser, Namespace
from macenko_mod import TorchMacenkoNormalizer
from torchvision import transforms
import torch
from typing import Literal



def get_subsample_rate(slide, mag_power=20):
    # Get the subsample rate compared to level 0
    # (necessary since levels in tcgaGBM data is not downsampled by the power of 2)
    mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    assert (
        mag >= mag_power
    ), f"Magnification of the slide ({mag}X) is smaller than the desired magnification ({mag_power}X)."

    ds_rate = mag / mag_power
    return int(ds_rate)

def get_sampling_params(slide, mag_power=20):
    # Get the optimal openslide level and subsample rate, given a magnification power
    mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    assert (
        mag >= mag_power
    ), f"Magnification of the slide ({mag}X) is smaller than the desired magnification ({mag_power}X)."
    ds_rate = mag / mag_power

    lvl_ds_rates = np.array(slide.level_downsamples).astype(np.int32)
    levels = np.arange(len(lvl_ds_rates))
    # Get levels that is larger than the given mag power
    idx_larger = np.argwhere(lvl_ds_rates <= ds_rate).flatten()
    lvl_ds_rates = lvl_ds_rates[idx_larger]
    levels = levels[idx_larger]
    # get the closest & larger mag power
    idx = np.argmax(lvl_ds_rates)
    closest_ds_rate = lvl_ds_rates[idx]
    opt_level = levels[idx]
    opt_ds_rate = ds_rate / closest_ds_rate

    return opt_level, opt_ds_rate

def read_region_by_power(slide, start, mag_power, width):
    opt_level, opt_ds_rate = get_sampling_params(slide, mag_power)
    read_width = tuple([int(opt_ds_rate * x) for x in width])
    im1 = slide.read_region(start, opt_level, read_width)
    if opt_ds_rate != 1:
        im1 = im1.resize(width, resample=Image.LINEAR)
    return im1

def get_thumbnail(WSI,mag_level=1.25):
    # thumbnail = mr_image.read_region((0, 0), max_level,mr_image.level_dimensions[max_level])

    xDim = WSI.level_dimensions[0][0]
    yDim = WSI.level_dimensions[0][1]
    mag = int(WSI.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    thumbnail_ds_rate = mag/mag_level
    thumnnail_size = (
        np.floor(xDim/thumbnail_ds_rate).astype(np.int64),
        np.floor(yDim/thumbnail_ds_rate).astype(np.int64),
    )
    thumbnail = read_region_by_power(
        WSI,
        (0,0),
        mag_level,
        thumnnail_size,
    )
    return np.array(thumbnail.convert("RGB"))

def fit_HE(wsi,mag_level=5,Io_source=240,beta=0.15):
    ## estimate the color normalization parameters for the given WSI, using thumbnails
    # inputs:
    #   wsi: an openslide object, or numpy array
    #   mag_level: the magnification level to use for the thumbnail
    #   Io_source: the transmitted light intensity for the source image
    #   beta: the transparency threshold for the color normalization
    # returns:
    #   torch_normalizer: a TorchMacenkoNormalizer object
    #   HE: the estimated Hematoxylin and Eosin color normalization matrixs
    if isinstance(wsi,openslide.OpenSlide):
        thumbnail = get_thumbnail(wsi,mag_level)
    elif isinstance(wsi,np.ndarray):
        thumbnail = wsi
    else:
        raise ValueError('wsi should be an openslide object or a numpy array')
    torch_normalizer = TorchMacenkoNormalizer()
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    device = torch.device("cpu")

    thumbnail_s = T(thumbnail).to(device)
    HE, _ = torch_normalizer.fit_source(thumbnail_s, Io=Io_source,beta=beta)
    return torch_normalizer, HE


# def score_tile(img,torch_normalizer, Io_source=240,Io_target=240,beta=0.15,HE_profile=None, method:Literal['cellularity','nuelci'] = 'cellularity'):
#     '''
#     Score a tile based on the color normalization
#     Inputs:
#         img: a numpy array of shape (H,W,3)
#         torch_normalizer: a TorchMacenkoNormalizer object
#         Io_source: the transmitted light intensity for the source image
#         Io_target: the transmitted light intensity for the target image
#         beta: the transparency threshold for the color normalization
#         HE_profile: the Hematoxylin and Eosin color normalization matrixs
#         method: the method to use for scoring the tile. Options are 'cellularity' and 'nuclei'
#     '''
#     T = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x*255)
#     ])
    
#     I_norm,H,E = torch_normalizer.normalize(
#         T(img) , Io=Io_source,Io_out=Io_target, beta = beta, HE=HE_profile,stains=True)