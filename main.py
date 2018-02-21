'''
pipeline:
    1. load data
    2. unpickle model
    3. apply model
    4. generate output masks
    5. create run-length encoded file`
    6. evaluate model
    7. submit kaggle
'''

import os
import glob
import skimage
import numpy as np

import subprocess

import utils.imaging
import utils.run_length_encoding
import utils.evaluate
from utils.imaging import label_mask, get_path, get_image_ids

from skimage.color import rgba2rgb, rgb2gray
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.segmentation import mark_boundaries
from skimage import morphology

def segment_image(image):
    if image.shape[2] == 4:
        image = rgba2rgb(image)
    image_gray = rgb2gray(image)
    thresh_val = threshold_otsu(image_gray)
    mask = thresh_val > image_gray
    mask = morphology.remove_small_holes(mask, min_size=20)
    return mask

def apply_segmentation_images(image_type='train', stage_num = 1):
    stage_num = str(stage_num)
    file_path = get_path('data_' + image_type + '_' + stage_num)
    image_ids = get_image_ids(file_path)
    output_path = get_path('output_' + image_type + '_' + stage_num + '_lab_seg')

    for idx, image_id in enumerate(image_ids):
        image_dir = file_path + image_id + "/images/" + \
                    image_id + ".png"
        image = skimage.io.imread(image_dir)
        mask = segment_image(image)
        labels = label_mask(mask)
        imsave(output_path + image_id + '.png', labels)
        print('saved image %d of %d, image: %s \n' % \
              (idx + 1, len(image_ids), image_id))


if __name__ == '__main__':
    apply_segmentation_images('train')
    apply_segmentation_images('test')
    utils.imaging.label_ground_truth_masks(1)
    utils.imaging.ground_truth_annotate(1)
    utils.imaging.segmented_annotate(image_type = 'train', stage_num = 1)
    utils.imaging.segmented_annotate(image_type = 'test', stage_num = 1)
    score = utils.evaluate.evaluate_images()
    df = utils.run_length_encoding.rle_images_in_dir(image_type = 'test', stage_num = 1)
