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

from utils import imaging

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

def apply_segmentation_images(image_type='train'):
    if image_type == 'train':
        data_dir = imaging.get_path('training_data')
        output_path = imaging.get_path('output') + image_type + "/labelled_segmented/"
    if image_type == 'test':
        data_dir = imaging.get_path('test_data')
        output_path = imaging.get_path('output') + image_type + "/labelled_segmented/"

    image_ids = imaging.get_image_ids(data_dir)

    for idx, image_id in enumerate(image_ids):
        image_dir = data_dir + image_id + "/images/" + \
                    image_id + ".png"
        image = skimage.io.imread(image_dir)
        mask = segment_image(image)
        labels = imaging.label_mask(mask)
        imsave(output_path + image_id + '.png', labels)
        print('saved image %d of %d, image: %s \n' % \
              (idx + 1, len(image_ids), image_id))


if __name__ == '__main__':
    apply_segmentation_images('train')
    apply_segmentation_images('test')
