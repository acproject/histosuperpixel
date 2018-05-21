# A sample CLI code for Raj and Mohamed, written by Sanghoon 5/1/2018
# input: a slide path
# output: superpixel information

import os
import sys

import numpy as np
import dask
import h5py

import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation as htk_seg

import histomicstk.utils as htk_utils

import large_image

from ctk_cli import CLIArgumentParser

import logging
logging.basicConfig(level=logging.CRITICAL)

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
from cli_common import utils as cli_utils  # noqa

from skimage.measure import regionprops
from skimage.segmentation import slic


def compute_superpixel_data(img_path, tile_position, wsi_mean,
                            wsi_stddev, args, **it_kwargs):

    # get slide tile source
    ts = large_image.getTileSource(img_path)

    # get requested tile information
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        resample=True,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **it_kwargs)

    im_tile = tile_info['tile'][:, :, :3]

    # get global x and y positions
    left = tile_info['gx']
    top = tile_info['gy']

    # get scale
    scale = tile_info['gwidth'] / tile_info['width']

    # perform color normalization
    im_nmzd = htk_cnorm.reinhard(im_tile,
                                 args.reference_mu_lab, args.reference_std_lab,
                                 wsi_mean, wsi_stddev)

    # compute the number of super-pixels
    im_width, im_height = im_nmzd.shape[:2]
    n_superpixels = (im_width/args.patchSize) * (im_height/args.patchSize)

    #
    # Generate labels using a superpixel algorithm (SLIC)
    # In SLIC, compactness controls image space proximity.
    # Higher compactness will make the shape of superpixels more square.
    #
    im_label = slic(im_nmzd, n_segments=n_superpixels,
                    compactness=args.compactness) + 1

    region_props = regionprops(im_label)

    # set superpixel data list
    s_data = []
    x_cent = []
    y_cent = []
    x_brs = []
    y_brs = []

    for i in range(len(region_props)):
        # get x, y centroids for superpixel
        cen_x, cen_y = region_props[i].centroid

        # get bounds of superpixel region
        min_row, max_row, min_col, max_col = \
            get_patch_bounds(cen_x, cen_y, args.patchSize, im_width, im_height)

        # grab superpixel label mask
        lmask = (
            im_label[:, :] == region_props[i].label).astype(np.bool)

        # embed with center pixel in middle of padded window
        emask = np.zeros((lmask.shape[0] + 2, lmask.shape[1] + 2), dtype=np.bool)
        emask[1:-1, 1:-1] = lmask

        # find boundaries
        bx, by = htk_seg.label.trace_object_boundaries(emask)

        with np.errstate(invalid='ignore'):
            # remove redundant points
            mby, mbx = htk_utils.merge_colinear(
                by[0].astype(float), bx[0].astype(float))

        scaled_x = (mbx - 1) * scale
        scaled_y = (mby - 1) * scale

        # get superpixel boundary at highest-res
        x_brs.append(scaled_x + top)
        y_brs.append(scaled_y + left)

        rgb_data = im_nmzd[min_row:max_row, min_col:max_col]

        s_data.append(rgb_data)

        # get superpixel centers at highest-res
        x_cent.append(
            round((cen_x * scale + top), 1))
        y_cent.append(
            round((cen_y * scale + left), 1))

    return s_data, x_cent, y_cent, x_brs, y_brs


def get_patch_bounds(cx, cy, patch_size, m, n):

    half_patch_size = patch_size/2.0

    min_row = int(round(cx) - half_patch_size)
    max_row = int(round(cx) + half_patch_size)
    min_col = int(round(cy) - half_patch_size)
    max_col = int(round(cy) + half_patch_size)

    if min_row < 0:
        max_row = max_row - min_row
        min_row = 0

    if max_row > m-1:
        min_row = min_row - (max_row - (m-1))
        max_row = m-1

    if min_col < 0:
        max_col = max_col - min_col
        min_col = 0

    if max_col > n-1:
        min_col = min_col - (max_col - (n-1))
        max_col = n-1

    return min_row, max_row, min_col, max_col


def main(args):  # noqa: C901

    # initiate dask client
    c = cli_utils.create_dask_client(args)

    # read input slide
    ts = large_image.getTileSource(args.inputSlidePath)

    # compute colorspace statistics (mean, variance) for whole slide
    wsi_mean, wsi_stddev = htk_cnorm.reinhard_stats(
        args.inputSlidePath, args.sample_fraction, args.analysis_mag)

    # compute tissue/foreground mask at low-res for whole slide images
    im_fgnd_mask_lres, fgnd_seg_scale = \
        cli_utils.segment_wsi_foreground_at_low_res(ts)

    # compute foreground fraction of tiles in parallel using Dask
    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
    }

    tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
        args.inputSlidePath, im_fgnd_mask_lres, fgnd_seg_scale,
        **it_kwargs
    )

    #
    # Now, we detect superpixel data in parallel using Dask
    #
    print('\n>> Detecting superpixel data ...\n')

    tile_result_list = []

    for tile in ts.tileIterator(**it_kwargs):

        tile_position = tile['tile_position']['position']

        if tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
            continue

        # detect superpixel data
        cur_result = dask.delayed(compute_superpixel_data)(
            args.inputSlidePath,
            tile_position,
            wsi_mean, wsi_stddev,
            args, **it_kwargs)

        # append result to list
        tile_result_list.append(cur_result)

    tile_result_list = dask.delayed(tile_result_list).compute()

    # initiate output data list
    superpixel_data = []
    x_centroids = []
    y_centroids = []
    x_boundaries = []
    y_boundaries = []

    for s_data, x_cent, y_cent, x_brs, y_brs in tile_result_list:

        for s_d in s_data:
            superpixel_data.append(s_d)

        for x_c in x_cent:
            x_centroids.append(x_c)

        for y_c in y_cent:
            y_centroids.append(y_c)

        for x_b in x_brs:
            x_boundaries.append(x_b)

        for y_b in y_brs:
            y_boundaries.append(y_b)

    superpixel_data = np.asarray(superpixel_data, dtype=np.float32)

    n_superpixels = len(superpixel_data)
    x_centroids = np.asarray(x_centroids).reshape((n_superpixels, 1))
    y_centroids = np.asarray(y_centroids).reshape((n_superpixels, 1))

    #
    # Last, we can store the data
    #
    print('>> Writing superpixel data information')

    output = h5py.File(args.outputSuperpixelFeatureFile, 'w')
    output.create_dataset('features', data=superpixel_data)
    output.create_dataset('x_centroid', data=x_centroids)
    output.create_dataset('y_centroid', data=y_centroids)
    output.close()

    #
    # Create Text file for boundaries
    #
    print('>> Writing text boundary file')

    boundary_file = open(args.outputBoundariesFile, 'w')

    for i in range(n_superpixels):
        boundary_file.write("%.1f\t" % y_centroids[i, 0])
        boundary_file.write("%.1f\t" % x_centroids[i, 0])

        for j in range(len(x_boundaries[i])):
            boundary_file.write(
                "%d,%d " % (y_boundaries[i][j], x_boundaries[i][j]))

        boundary_file.write("\n")


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())