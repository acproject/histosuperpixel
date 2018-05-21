# A sample CLI code for Raj and Mohamed, written by Sanghoon 5/3/2018
# input: a slide path
# output: superpixel information
# Modified and extracted code from Juypter Notebook to run on terminal


import dask
import sys
sys.path.append('../cli_common/')
import utils as cli_utils
import histomicstk.preprocessing.color_normalization as htk_cnorm
import large_image
import histomicstk.utils as htk_utils
from skimage.measure import regionprops
from skimage.segmentation import slic
import numpy as np
import h5py
from ctk_cli import CLIArgumentParser
import dask.distributed as dd
import logging
logging.basicConfig(level=logging.CRITICAL)

def create_dask_client():
    """Create and install a Dask distributed client using args from a
    Namespace, supporting the following attributes:
    - .scheduler_address: Address of the distributed scheduler, or the
      empty string to start one locally

    """
    #  scheduler = dd.LocalCluster(scheduler_port=8786)

    

    # scheduler_address = dask.distributed.LocalCluster(
    #     ip='0.0.0.0',  # Allow reaching the diagnostics port externally
    #     scheduler_port=0,  # Don't expose the scheduler port
    #     silence_logs=False
    # )
    
    # return dask.distributed.Client('127.0.0.1:3000')
    return dask.distributed.Client()

#sample_fraction = 0.1
#analysis_mag = 10
#ts = large_image.getTileSource(slidePath)
# compute colorspace statistics (mean, variance) for whole slide
#wsi_mean, wsi_stddev = htk_cnorm.reinhard_stats(slidePath, sample_fraction, analysis_mag)

# compute tissue/foreground mask at low-res for whole slide images
#im_fgnd_mask_lres, fgnd_seg_scale = cli_utils.segment_wsi_foreground_at_low_res(ts)

# compute foreground fraction of tiles in parallel using Dask
#analysis_tile_size = 2048
#analysis_mag = 10
#it_kwargs = {
#    'tile_size': {'width': analysis_tile_size},
#    'scale': {'magnification': analysis_mag},
#}

#inputSlidePath = 0

#tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
#    slidePath, im_fgnd_mask_lres, fgnd_seg_scale,
#    **it_kwargs
#)

#print('\n>> Detecting superpixel data ...\n')


def compute_superpixel_data(img_path, tile_position, wsi_mean, wsi_stddev):
    
    # get slide tile source
    ts = large_image.getTileSource(img_path)

    # get requested tile information
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        resample=True,
        format=large_image.tilesource.TILE_FORMAT_NUMPY)


    im_tile = tile_info['tile'][:, :, :3]

     # get global x and y positions
    left = tile_info['gx']
    top = tile_info['gy']

    # get scale
    scale = tile_info['gwidth'] / tile_info['width']


    reference_mu_lab = [8.63234435, -0.11501964, 0.03868433]
    reference_std_lab = [0.57506023, 0.10403329, 0.01364062]

    # perform color normalization
    im_nmzd = htk_cnorm.reinhard(im_tile,
                                 reference_mu_lab, reference_std_lab,
                                 wsi_mean, wsi_stddev)
    patchSize = 32
    # compute the number of super-pixels
    im_width, im_height = im_nmzd.shape[:2]
    n_superpixels = (im_width/patchSize) * (im_height/patchSize)

    #
    # Generate labels using a superpixel algorithm (SLIC)
    # In SLIC, compactness controls image space proximity.
    # Higher compactness will make the shape of superpixels more square.
    #

    compactness = 50
    im_label = slic(im_nmzd, n_segments=n_superpixels,
                    compactness=compactness) + 1

    region_props = regionprops(im_label)

    # set superpixel data list
    s_data = []
    x_cent = []
    y_cent = []

    for i in range(len(region_props)):
        # get x, y centroids for superpixel
        cen_x, cen_y = region_props[i].centroid

        # get bounds of superpixel region
        min_row, max_row, min_col, max_col = \
            get_patch_bounds(cen_x, cen_y, patchSize, im_width, im_height)

        rgb_data = im_nmzd[min_row:max_row, min_col:max_col]

        s_data.append(rgb_data)

         # get superpixel centers at highest-res
        x_cent.append(
            round((cen_x * scale + top), 1))
        y_cent.append(
            round((cen_y * scale + left), 1))

    return s_data, x_cent, y_cent

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

    # inputSlidePath = 'test2_superfixel.svs'
    # outputSuperpixelFeatureFile= 
    #scheduler = dd.LocalCluster(scheduler_port=2222)
    c = create_dask_client() 
    print('\n>> Creating Dask client and printing its values...\n')
    #print c

    ts = large_image.getTileSource(args.inputSlidePath)

    sample_fraction = 0.1
    analysis_mag = 10
    #ts = large_image.getTileSource(slidePath)
    # compute colorspace statistics (mean, variance) for whole slide
    wsi_mean, wsi_stddev = htk_cnorm.reinhard_stats(args.inputSlidePath, sample_fraction, analysis_mag)

    # compute tissue/foreground mask at low-res for whole slide images
    im_fgnd_mask_lres, fgnd_seg_scale = cli_utils.segment_wsi_foreground_at_low_res(ts)

    # compute foreground fraction of tiles in parallel using Dask
    analysis_tile_size = 2048
    analysis_mag = 10
    it_kwargs = {
        'tile_size': {'width': analysis_tile_size},
        'scale': {'magnification': analysis_mag},
    }

    inputSlidePath = 0

    tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
        args.inputSlidePath, im_fgnd_mask_lres, fgnd_seg_scale,
        **it_kwargs
    )


    tile_result_list = []
    min_fgnd_frac = 0.001

    for tile in ts.tileIterator(**it_kwargs):
        tile_position = tile['tile_position']['position']
        if tile_fgnd_frac_list[tile_position] <= min_fgnd_frac:
            continue
        #tile_result_list.append(compute_superpixel_data(args.inputSlidePath, tile_position, wsi_mean, wsi_stddev))
        # detect superpixel data
        cur_result = dask.delayed(compute_superpixel_data)(
            "test.svs",
            tile_position,
            wsi_mean, wsi_stddev)

        # append result to list
        tile_result_list.append(cur_result)
	print 'hello'

    tile_result_list = dask.delayed(tile_result_list).compute()

    # initiate output data list
    superpixel_data = []
    x_centroids = []
    y_centroids = []


    for s_data, x_cent, y_cent in tile_result_list:

        for s_d in s_data:
            superpixel_data.append(s_d)

        for x_c in x_cent:
            x_centroids.append(x_c)

        for y_c in y_cent:
            y_centroids.append(y_c)

    superpixel_data = np.asarray(superpixel_data, dtype=np.float32)


    n_superpixels = len(superpixel_data)
    x_centroids = np.asarray(x_centroids).reshape((n_superpixels, 1))
    y_centroids = np.asarray(y_centroids).reshape((n_superpixels, 1))

    print('>> Writing superpixel data information')

    # output = h5py.File('superpixelResults1', 'w')
    output = h5py.File(args.outputSuperpixelFeatureFile, 'w')
    output.create_dataset('features', data=superpixel_data)
    output.create_dataset('x_centroid', data=x_centroids)
    output.create_dataset('y_centroid', data=y_centroids)
    output.close()

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
