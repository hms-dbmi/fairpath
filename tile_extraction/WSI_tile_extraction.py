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

import pandas as pd
# from utils import fit_HE
from histolab.scorer import CellularityScorer, NucleiScorer
from histolab.tile import Tile

TQDM_INVERVAL = 1
THUMBNAIL_MAG_LEVEL = 1.25
# read params from JSON file
# BLOB_MAG_LEVEL = 5


def rgbHueSat(pix):
    # pixImg=Image.fromarray(pix)

    pixHsv = pix.convert("HSV")
    pixHsvArray = np.array(pixHsv, dtype=np.float32)
    return pixHsvArray[:, :, 0], pixHsvArray[:, :, 1]


def GrayscaleOpticalDensity(I, Io=240):
    # I: np array
    # I_gray = ImageOps.grayscale(I)
    OD = -np.log((I[:, :, :3].astype(np.float32) + 1) / Io)
    OD_gray = rgb2gray(OD)

    return OD_gray


def OpticalDensityThreshold(I, Io=240, beta=0.15):
    # # calculate optical density
    # OD = -np.log((I.astype(np.float32) + 1) / Io)
    # # remove transparent pixels
    # ODhat = ~np.any(OD < beta, axis=2)

    OD_gray = GrayscaleOpticalDensity(I,Io)
    ODhat = OD_gray > beta

    return OD_gray, ODhat





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


class ThumbnailTileVisualizer:
    '''
    A class to visualize the valid tiles on the thumbnail of a WSI
    '''

    def __init__(self, thumbnail, grid_size, grid_stride, bg_scale=0.8):
        self.thumbnail = np.array(thumbnail)[:,:,:3]
        self.grid_size = grid_size
        self.grid_stride = grid_stride
        self.bg_scale = bg_scale
        self.init_thumbnail_mask()

    def init_thumbnail_mask(self):
        # create grid image
        thumbnail_mask = np.copy(self.thumbnail)

        for i in range(3):
            thumbnail_mask[:, :, i] = self.bg_scale * thumbnail_mask[:, :, i]
        self.thumbnail_mask = thumbnail_mask

    def highlight_tile(self, loc):
        '''
        loc: tile index in X & Y (Tuple)
        '''
        loc = list(loc)
        i_start = [int(a * b)-1 for a, b in zip(self.grid_stride, loc)]
        i_end = [int(a * b)+int(c)+2 for a, b,
                 c in zip(self.grid_stride, loc, self.grid_size)]
        # self.thumbnail_mask[i_start[1]:i_end[1],i_start[0]:i_end[0],3] = 1
        tile = np.copy(self.thumbnail[i_start[1]:i_end[1], i_start[0]:i_end[0], :])
        ## create black border
        tile[0:2,:,:] = 0
        tile[-2:,:,:] = 0
        tile[:,0:2,:] = 0
        tile[:,-2:,:] = 0
        self.thumbnail_mask[i_start[1]:i_end[1], i_start[0]:i_end[0], :] = tile
        

        # for i in range(3):
        #     self.thumbnail_mask[i_start[1]:i_end[1], i_start[0]:i_end[0],
        #                         i] = self.thumbnail[i_start[1]:i_end[1], i_start[0]:i_end[0], i]
        return

    def save_overlay(self, filename):
        overlay = Image.fromarray(self.thumbnail_mask)
        overlay.convert("RGB").save(filename)


def tile_WSI(params):
    # for filename in os.listdir(dirname):

    # locals().update(JsoncParser.parse_file(args.params))
    tile_params = JsoncParser.parse_file(params.params)
    print("=======  Tiling Parameters:  =======")
    for key in tile_params.keys():
        print(f"  {key}:\t{tile_params[key]}")
    tile_params = Namespace(**tile_params)

    if not hasattr(tile_params, 'useBlobDetection'):
        tile_params.useBlobDetection = False

    tile_params.hueLowerBound = tile_params.hueLowerBound / 360 * 255
    tile_params.hueUpperBound = tile_params.hueUpperBound / 360 * 255
    tile_params.saturationLowerBound = tile_params.saturationLowerBound * 255
    ###########################
    print("=======      End      =======")
    TQDM_INVERVAL = 1

    if params.debug:
        tile_params.xStride = 1000
        tile_params.yStride = 1000

        tile_params.xStride = 256
        tile_params.yStride = 256
        TQDM_INVERVAL = 1
        params.overwrite_if_exist = True
        # params.infile = "/n/data2/hms/dbmi/kyu/lab/datasets/tcgaGBM/svs/b1988b0e-2122-418d-a9ff-a0e9c566ebc5/TCGA-06-0216-01A-01-BS1.076e2c52-01ca-48bc-b93c-e916bf64718b.svs"
        # params.infile = "/n/data2/hms/dbmi/kyu/lab/jz290/EbrainData/PCNSL/a1951f04-357f-11eb-a8df-001a7dda7111.ndpi"
        # params.infile = "/n/data2/hms/dbmi/kyu/lab/jz290/EbrainData/PCNSL/a194f790-357f-11eb-b605-001a7dda7111.ndpi"
        # params.infile = "/n/data2/hms/dbmi/kyu/lab/datasets/GBMFrozen/ndpi/BD19-0013/A04_T90/A04_T90_BD19-0013-1_2019-09-18_00.37.07.ndpi"
        # params.infile = "/n/data2/hms/dbmi/kyu/lab/datasets/tcgaLUAD/newSvs/3a5b6121-72bd-4f07-a195-a914eb6060ec/TCGA-50-8459-01A-01-TS1.ccd667c6-cc96-43a9-9afc-4f1c23935ffc.svs"
        # params.infile = "/n/data2/hms/dbmi/kyu/lab/jz290/Ebrains-Control/86242949-7775-11eb-9c4b-001a7dda7111.ndpi"
        params.infile = "/n/data2/hms/dbmi/kyu/lab/datasets/DFCI_breast/raw/674161/BL-15-J23873/9957.svs"
        params.infile = '/n/data2/hms/dbmi/kyu/lab/datasets/DFCI_breast/raw/242863/BL-15-N42366/11625.svs'

    filename = os.path.basename(params.infile)

    outputDirBase = os.path.join(
        params.outpath,
        f"{params.proj}_{tile_params.xPatch}Stride{tile_params.xStride}_max{tile_params.numPatches}_Q{tile_params.top_Q}_Zoom{tile_params.IMG_MAG_LEVEL}X/",
    )

    if not os.path.isdir(outputDirBase):
        try:
            os.makedirs(outputDirBase)
        except:
            pass

    outputDir = os.path.join(outputDirBase, filename)
    outputDirStats = os.path.join(outputDirBase, "tileStats/")
    outputDirThumbNail = os.path.join(outputDirBase, "thumbnail/")
    outputDirOverlay = os.path.join(outputDirBase, "overlay_vis/")
    outputDirOverlay2 = os.path.join(outputDirBase, f"overlay_vis_top{tile_params.numPatches}/")
    AllValidTallyFileName = os.path.join(
        outputDirStats, f"AllValidSatTally{filename}.csv"
    )
    mr_image = openslide.OpenSlide(params.infile)
    # level = get_zoom_level(mr_image, mag_power=tile_params.IMG_MAG_LEVEL)
    # search_level = get_zoom_level(mr_image, mag_power=tile_params.SEARCH_MAG_LEVEL)
    ds_rate = get_subsample_rate(mr_image, mag_power=tile_params.IMG_MAG_LEVEL)
    search_ds_rate = get_subsample_rate(
        mr_image, mag_power=tile_params.SEARCH_MAG_LEVEL)
    blob_ds_rate = get_subsample_rate(
        mr_image, mag_power=tile_params.BLOB_MAG_LEVEL)

    if os.path.isfile(AllValidTallyFileName):
        if not params.overwrite_if_exist:
            raise FileExistsError(
                f"TileStats {os.path.basename(AllValidTallyFileName)} already exists. Exited for safety reason (consider saving in another root directory or setting --overwrite_if_exist to True )"
            )
        else:
            print(
                f"Directory {os.path.basename(AllValidTallyFileName)} already exists. Will overwrite the existing file."
            )

    os.makedirs(outputDirStats, exist_ok=True)
    os.makedirs(outputDirThumbNail, exist_ok=True)
    os.makedirs(outputDirOverlay, exist_ok=True)
    os.makedirs(outputDirOverlay2, exist_ok=True)

    lvl_xStride = tile_params.xStride * ds_rate
    lvl_yStride = tile_params.yStride * ds_rate
    # ds = mr_image.getLevelDownsample(level)
    xDim = mr_image.level_dimensions[0][0]
    yDim = mr_image.level_dimensions[0][1]

    AllValidTally = []
    nonAllCriteriaSumCounts = []
    blobsCount = []

    #
    nonEmpty = 0

    # if params.debug:
    ##
    # Save thumbnail
    max_level = mr_image.level_count - 1
    # thumbnail = mr_image.read_region((0, 0), max_level,mr_image.level_dimensions[max_level])
    mag = int(mr_image.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    thumbnail_ds_rate = mag/THUMBNAIL_MAG_LEVEL
    thumnnail_size = (
        int(np.floor(xDim/thumbnail_ds_rate)),
        int(np.floor(yDim/thumbnail_ds_rate)),
    )
    thumbnail = read_region_by_power(
        mr_image,
        (0, 0),
        THUMBNAIL_MAG_LEVEL,
        thumnnail_size,
    )
    # thumbnail_size = 1000
    # thumbnail = mr_image.get_thumbnail((thumbnail_size, thumbnail_size))
    thumbnail_gridsize = (
        tile_params.xPatch/(tile_params.IMG_MAG_LEVEL/THUMBNAIL_MAG_LEVEL),
        tile_params.yPatch/(tile_params.IMG_MAG_LEVEL/THUMBNAIL_MAG_LEVEL))

    thumbnail_gridstride = (
        tile_params.xStride/(tile_params.IMG_MAG_LEVEL/THUMBNAIL_MAG_LEVEL),
        tile_params.yStride/(tile_params.IMG_MAG_LEVEL/THUMBNAIL_MAG_LEVEL))
    thumbnail_visualizer = ThumbnailTileVisualizer(
        thumbnail, thumbnail_gridsize, thumbnail_gridstride)
    
    thumbnail_visualizer_selected = ThumbnailTileVisualizer(
        thumbnail, thumbnail_gridsize, thumbnail_gridstride)
    thumbnail_name = os.path.join(outputDirThumbNail, f"{filename}.jpg")
    thumbnail.convert("RGB").save(thumbnail_name)
    # and np.sqrt(np.mean(OD_LoG**2)) > tile_params.LOGLowerBound:
    if params.debug:
        thumbnail.convert("RGB").save("thumbnail.jpg")

    ##
    # If OD_use_Otsu == True, calculate intensity threhsold (beta) from the thumbnail
    if hasattr(tile_params, "OD_use_Otsu"):
        if tile_params.OD_use_Otsu == True:
            print("Calulating Optical Density threshold using Otsu's Method:")
            OD_thumbnail = GrayscaleOpticalDensity(
                np.array(thumbnail),  Io=tile_params.OD_Io)

            OD_threshs = threshold_multiotsu(OD_thumbnail)
            OD_BG_thresh = OD_threshs[0]
            tile_params.OD_beta = OD_BG_thresh
            print("OD_use_Otsu set to True. Calculating OD_beta using Otsu's method...")
            print(f"OD_beta set to {OD_BG_thresh}")
            if params.debug:
                plt.close()
                # plt.hist(OD_thumbnail.flatten(),bins = 100)
                counts, bins = np.histogram(OD_thumbnail.flatten(), bins=100)
                plt.stairs(counts, bins)
                for OD_thresh in OD_threshs:
                    plt.plot([OD_thresh, OD_thresh], [0, np.max(counts)], ':r')
                plt.savefig('test_otsu.jpg')
                plt.close()
    if hasattr(tile_params, "useCellScorer"):
        if tile_params.useCellScorer == True:
            # print("Cell Scorer is used. Calculating color profile using Macenko's method...")
            # normalizer, HE =  fit_HE(thumbnail,mag_level=5,Io_source=tile_params.OD_Io,beta=tile_params.OD_beta)
            if tile_params.CellScorer == 'CellularityScorer':
                # set consider_tissue to False because efficiency (we already calculated tissue mask from our side)
                cell_scorer = CellularityScorer(consider_tissue=False)
            elif tile_params.CellScorer == 'NucleiScorer':
                cell_scorer = NucleiScorer()
            else:
                raise ValueError(f"CellScorer {tile_params.CellScorer} not recognized")
    else:
        tile_params.useCellScorer = False
    
    if hasattr(tile_params, "sat_use_Otsu"):
        if tile_params.sat_use_Otsu == True:
            print("Calulating Saturation threshold using Otsu's Method:")

            pixHue, pixSat = rgbHueSat(thumbnail)

            sat_threshs = threshold_multiotsu(pixSat.astype(np.float32))
            sat_thresh = sat_threshs[0]
            tile_params.saturationLowerBound = sat_thresh

            print("sat_use_Otsu set to True. Calculating OD_beta using Otsu's method...")
            print(f"saturationLowerBound set to {sat_thresh}")

        
    N_tiles_est = int(xDim / lvl_xStride) * int(yDim / lvl_yStride)
    print(f"Estimated total number of tiles to check: {N_tiles_est}")
    pbar = tqdm(range(int(xDim / lvl_xStride)), mininterval=TQDM_INVERVAL)
    count = 0
    sub_size = (
        int(tile_params.xPatch / search_ds_rate * ds_rate),
        int(tile_params.yPatch / search_ds_rate * ds_rate),
    )

    blob_sub_size = (
        int(tile_params.xPatch / blob_ds_rate * ds_rate),
        int(tile_params.yPatch / blob_ds_rate * ds_rate),
    )

    # if thresh = threshold_otsu(image):

    count_threshold = tile_params.area_threhold * sub_size[0] * sub_size[1]
    nonBlack_count_threshold = (
        1 - tile_params.BlackAreaThreshold) * sub_size[0] * sub_size[1]

    AllValidTallyList = []
    if params.read_tile_csv:
        # read and existing tile csv file
        # df_AllValid = pd.read_csv(params.read_tile_csv)
        df_allValid = pd.read_csv(AllValidTallyFileName)
    else:
        # get tile csv file with tile detection
        for i in pbar:
            # print(f"{i}/{int(xDim/lvl_xStride)}")
            pbar.set_description(
                f" {len(AllValidTallyList)} Valid, {count} Total",refresh=False)
            for j in range(int(yDim / lvl_yStride)):
                count = count + 1
                # print(i,j)
                # try:
                image_patch = read_region_by_power(
                    mr_image,
                    (lvl_xStride * i, lvl_yStride * j),
                    tile_params.SEARCH_MAG_LEVEL,
                    sub_size,
                )

                pix = np.array(image_patch)
                pix = pix[:, :, 0:3]
                
                # Optical Density
                OD_gray, nonEmpty = OpticalDensityThreshold(
                    pix, Io=tile_params.OD_Io, beta=tile_params.OD_beta)
                # nonEmpty = np.logical_and(OD_gray > OD_threshs[0], OD_gray < OD_threshs[1])
                nonEmpty = np.where(nonEmpty, 1, 0)
                nonEmptySum = nonEmpty.sum()
                if nonEmptySum < count_threshold:
                    continue
                
                # log
                if tile_params.LOGLowerBound > 0:
                    # to save time, only run LoG filter is tile_params.LOGLowerBound > 0
                    OD_LoG = ndimage.gaussian_laplace(
                        OD_gray, sigma=tile_params.LOGsigma)
                    OD_abs_LoG = np.abs(OD_LoG)
                else:
                    OD_abs_LoG = np.ones_like(OD_gray)
                nonLoG = np.where(
                    OD_abs_LoG > tile_params.LOGLowerBound, 1, 0)
                nonLoGSum = np.sum(nonLoG)
                if nonLoGSum < count_threshold:
                    continue
                ## Sat and hue
                pixHue, pixSat = rgbHueSat(image_patch)
                nonHue = np.where(np.logical_or(
                    pixHue < tile_params.hueLowerBound, pixHue >= tile_params.hueUpperBound), 1, 0)
                nonSat = np.where(
                    pixSat > tile_params.saturationLowerBound, 1, 0)
                nonHueSum = np.sum(nonHue)
                if nonHueSum < count_threshold:
                    continue
                nonSatSum = np.sum(nonSat)
                if nonSatSum < count_threshold:
                    continue
                # Black
                nonBlack = np.where(np.max(pix, axis=2)
                                    > tile_params.BlackThresold, 1, 0)
                nonBlackSum = np.sum(nonBlack)
                if nonBlackSum < nonBlack_count_threshold:
                    continue
                ##
                    
                
                # nonLoG = np.where(OD_LoG < -tile_params.LOGLowerBound,1,0)
                nonAllCriteria = nonEmpty * nonHue * nonSat * nonBlack * nonLoG

                nonAllCriteriaSum = np.sum(nonAllCriteria)
                # nonAllCriteriaSum = np.min(nonAllCriteriaSum,nonBlackSum)

                # nonAllCriteria =  np.logical_and(np.logical_and(nonEmpty, nonHue),nonSat)

                # and np.sqrt(np.mean(OD_LoG**2)) > tile_params.LOGLowerBound:
                if params.debug:
                    image_patch.convert("RGB").save("test_patch.jpg")

                if (
                    nonAllCriteriaSum > count_threshold
                    and nonBlackSum > nonBlack_count_threshold
                ):
                    if tile_params.useCellScorer == True:
                        #  cellularity scoring
                        tile = Tile(Image.fromarray(pix), (0, 0))
                        score = cell_scorer(tile)
                        # normalize by tissue ratio
                        # if tile_params.CellScorer == 'CellularityScorer':
                        #     tissue_ratio = nonAllCriteriaSum/nonAllCriteria.size
                        #     score = score / tissue_ratio
                        ##
                        
                        AllValidTallyList.append(
                            {
                                "i": i,
                                "j": j,
                                "X": int(i * lvl_xStride),
                                "Y": int(j * lvl_yStride),
                                "mag_level": tile_params.IMG_MAG_LEVEL,
                                "width": tile_params.xPatch,
                                "height": tile_params.yPatch,
                                "BlobCount": score,
                                "nonAllCriteriaSum": nonAllCriteriaSum,
                                "nonHueSum": nonHueSum,
                                "nonSatSum": nonSatSum,
                                "nonLoGSum": nonLoGSum,
                                "nonBlackSum": nonBlackSum,
                            }
                        )
                        
                        thumbnail_visualizer.highlight_tile((i, j))
                        nonAllCriteriaSumCounts.append(
                            int(nonAllCriteriaSum))
                        blobsCount.append(score)
                    
                    elif tile_params.useBlobDetection:
                        # read higher-res image for blob detection
                        if tile_params.BLOB_MAG_LEVEL != tile_params.SEARCH_MAG_LEVEL:

                            image_patch = read_region_by_power(
                                mr_image,
                                (lvl_xStride * i, lvl_yStride * j),
                                tile_params.BLOB_MAG_LEVEL,
                                blob_sub_size,
                            )
                            pix = np.array(image_patch)
                            pix = pix[:, :, 0:3]
                            OD_gray, _ = OpticalDensityThreshold(
                                pix, Io=tile_params.OD_Io, beta=tile_params.OD_beta)

                        if params.blob_fun == 'log':
                            blobs = blob_log(
                                OD_gray,
                                min_sigma=tile_params.blobsRadiusLowerBound,
                                max_sigma=tile_params.blobsRadiusUpperBound,
                                num_sigma=tile_params.blobNumSigma,
                                threshold=tile_params.blobThreshold,
                            )
                        elif params.blob_fun == 'dog':
                            blobs = blob_dog(
                                OD_gray,
                                min_sigma=tile_params.blobsRadiusLowerBound,
                                max_sigma=tile_params.blobsRadiusUpperBound,
                                threshold=tile_params.blobThreshold,
                            )
                        elif params.blob_fun == 'doh':
                            blobs = blob_doh(
                                OD_gray,
                                min_sigma=tile_params.blobsRadiusLowerBound,
                                max_sigma=tile_params.blobsRadiusUpperBound,
                                num_sigma=tile_params.blobNumSigma,
                                threshold=tile_params.blobThreshold,
                            )
                        blobs = blobs[
                            np.logical_and(
                                blobs[:, 2] < tile_params.blobsRadiusUpperBound,
                                blobs[:, 2] > tile_params.blobsRadiusLowerBound,
                            ),
                            :,
                        ]
                        if tile_params.BLOB_ONLY_IN_VALID:
                            idx_valid = np.argwhere(
                                np.array(
                                    [
                                        nonAllCriteria[int(x), int(y)]
                                        for x, y in zip(blobs[:, 0], blobs[:, 1])
                                    ]
                                )
                            ).flatten()
                            blobs = blobs[idx_valid, :]

                        if blobs.shape[0] > tile_params.blobsNumLowerBound:

                            # AllValidTallyFile.write(str(i*lvl_xStride) + "\t" + str(j*lvl_yStride) + "\t" + str(nonEmptySum) + "\t" + str(nonHueSum) + "\t" + str(nonAllCriteriaSum) + "\n")
                            AllValidTallyList.append(
                                {
                                    "i": i,
                                    "j": j,
                                    "X": i * lvl_xStride,
                                    "Y": j * lvl_yStride,
                                    "mag_level": tile_params.IMG_MAG_LEVEL,
                                    "width": tile_params.xPatch,
                                    "height": tile_params.yPatch,
                                    "BlobCount": blobs.shape[0],
                                    "nonAllCriteriaSum": nonAllCriteriaSum,
                                    "nonHueSum": nonHueSum,
                                    "nonSatSum": nonSatSum,
                                    "nonLoGSum": nonLoGSum,
                                    "nonBlackSum": nonBlackSum,
                                }
                            )
                            thumbnail_visualizer.highlight_tile((i, j))
                            nonAllCriteriaSumCounts.append(
                                int(nonAllCriteriaSum))
                            blobsCount.append(blobs.shape[0])
                    
                    else:
                        # AllValidTallyFile.write(str(i*lvl_xStride) + "\t" + str(j*lvl_yStride) + "\t" + str(nonEmptySum) + "\t" + str(nonHueSum) + "\t" + str(nonAllCriteriaSum) + "\n")
                        AllValidTallyList.append(
                            {
                                "i": i,
                                "j": j,
                                "X": int(i * lvl_xStride),
                                "Y": int(j * lvl_yStride),
                                "mag_level": tile_params.IMG_MAG_LEVEL,
                                "width": tile_params.xPatch,
                                "height": tile_params.yPatch,
                                "BlobCount": 1,
                                "nonAllCriteriaSum": nonAllCriteriaSum,
                                "nonHueSum": nonHueSum,
                                "nonSatSum": nonSatSum,
                                "nonLoGSum": nonLoGSum,
                                "nonBlackSum": nonBlackSum,
                            }
                        )
                        thumbnail_visualizer.highlight_tile((i, j))
                        nonAllCriteriaSumCounts.append(
                            int(nonAllCriteriaSum))
                        blobsCount.append(1)
                # except:
                #     pix = 0
        df_AllValid = pd.DataFrame.from_records(AllValidTallyList)
        total_tiles = int(xDim / lvl_xStride) * int(yDim / lvl_yStride)
        nonAllCriteriaSumCounts = np.array(nonAllCriteriaSumCounts)

        blobsCount = np.array(blobsCount)

        print(f"Total Tiles: {total_tiles}")
        print(f"Nonempty Tiles: {df_AllValid.shape[0]}")

        SortVal = nonAllCriteriaSumCounts* blobsCount
        SortVal = SortVal/np.std(SortVal)
        df_AllValid['rankVal'] = SortVal
        df_AllValid =df_AllValid.sort_values('rankVal',ascending=False).reset_index(drop=True)

        df_AllValid["rank"] = np.arange(1, df_AllValid.shape[0] + 1)
        df_AllValid.to_csv(AllValidTallyFileName)
        ##
        overlay_name = os.path.join(outputDirOverlay, f"{filename}.jpg")
        thumbnail_visualizer.save_overlay(overlay_name)
        if params.debug:
            thumbnail_visualizer.save_overlay('test_overlay.jpg')

    nSelected = min(int(tile_params.numPatches), int(
        df_AllValid.shape[0] * tile_params.top_Q))
    ### outputting another thumbnail with selected tiles


    for i_row in tqdm(range(nSelected), mininterval=TQDM_INVERVAL):
        # for i in tqdm(range(len(AllValidTally)),mininterval=TQDM_INVERVAL):
        # print(i)
        
        row = df_AllValid.iloc[i_row]
        i = row['i']
        j = row['j']
        thumbnail_visualizer_selected.highlight_tile((i, j))
    
    overlay_name = os.path.join(outputDirOverlay2, f"{filename}.jpg")
    thumbnail_visualizer_selected.save_overlay(overlay_name)
    ###

    # nStart=0
    if params.output_tiles:
        print("Extracting Tiles...")
        os.makedirs(outputDir, exist_ok=True)
        for i in tqdm(range(nSelected), mininterval=TQDM_INVERVAL):
            # for i in tqdm(range(len(AllValidTally)),mininterval=TQDM_INVERVAL):
            # print(i)
            row = df_AllValid.iloc[i]
            outputFileName = os.path.join(
                outputDir,
                f"tile{i}_{filename}_{row['X']}_{row['Y']}.jpg"
            )
            image_patch = read_region_by_power(
                mr_image,
                (int(row['X']), int(row['Y'])),
                row["mag_level"],
                (row['width'], row['height']),
            )
            pix = np.array(image_patch)
            pix = pix[:, :, 0:3]
            Image.fromarray(pix).save(outputFileName)
    print("done")


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument(
        "--infile",
        help="""Input WSI File """,
        type=str,
        # default="/n/data2/hms/dbmi/kyu/lab/datasets/DFCI_breast/raw/674161/BL-15-J23873/9957.svs"
        default="/n/data2/hms/dbmi/kyu/lab/datasets/DFCI_breast/raw/113028/BL-16-X42771/16950.svs"
    )

    parser.add_argument(
        "--params",
        help="""Param File """,
        type=str,
        # default="image_patching_script/tile_params_default500.jsonc",
        default='/n/data2/hms/dbmi/kyu/lab/shl968/fairness_external_validation/step1_tile_extraction/tiling_params/tile_params_w512s512.jsonc'
    )

    parser.add_argument(
        "--outpath",
        help="""Output Path """,
        type=str,
        default="/n/data2/hms/dbmi/kyu/lab/shl968/tile_datasets/",
    )
    parser.add_argument(
        "--proj", help="""Project name (folder prefix) """, type=str, default="Debug")
    
    parser.add_argument(
        "--blob_fun", help="""Blob function  """, type=str, default="log", choices=["log", 'dog', 'doh']
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
        default=True,
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
    for key, val in vars(args).items():
        print(f'{key}:\t{val}')

    tile_WSI(args)
