import os
import glob
import skimage

import numpy as np
from scipy import ndimage
from skimage.color import rgba2rgb
from skimage.io import imread, imsave
from skimage.segmentation import mark_boundaries

def get_training_data_path():
    "gets the path to the training data directory"
    file_path = os.environ['DATA_FOLDER'] + 'stage1_train/'
    return file_path

def get_save_path():
    "gets the path to output images"
    save_path = os.environ['OUTPUT_FOLDER']
    return save_path

def get_image_ids(path):
    "returns a list of images at path"
    image_ids = sorted([f for f in os.listdir(path) \
                        if not f.startswith('.')])
    return image_ids

def label_mask(mask):
    ''' mask out backgroundm extract connected objects and label'''
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    return labels

def label_ground_truth_masks():
    ''' labels masks of training data and saves'''
    file_path = get_training_data_path()
    image_ids = get_image_ids(file_path)
    save_path = get_save_path() + "/labelled_ground_truth/"

    for idx, image_id in enumerate(image_ids):
        masks = file_path + image_id +  "/masks/*.png"
        masks = skimage.io.imread_collection(masks).concatenate()

        num_masks, height, width = masks.shape

        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1

        imsave(save_path + image_id + '.png', labels)

        print('saved image %d of %d, image: %s \n' % \
              (idx + 1, len(image_ids), image_id)) 

def ground_truth_annotate():
    "annotates images by showing ground truth contours from masks"
    file_path = get_training_data_path()
    image_ids = get_image_ids(file_path)
    save_path = get_save_path() +  "/annotated_ground_truth/"

    for idx, image_id in enumerate(image_ids):
        image_dir = file_path + image_id + "/images/" + \
                    image_id + ".png"
        image = skimage.io.imread(image_dir)
        image = rgba2rgb(image)

        masks = os.listdir(file_path + image_id + "/masks/")

        image_overlay = image
        for mask in masks:
            mask_name = file_path + image_id + "/masks/" + mask
            mask = skimage.io.imread(mask_name)
            image_overlay = mark_boundaries(image_overlay,
                                            mask, color=(0.6,0,0),
                                            outline_color=None,
                                            mode='outer')

        imsave(save_path + image_id + '.png', image_overlay)

        print('saved image %d of %d, image: %s \n' % \
              (idx + 1, len(image_ids), image_id))

if __name__ == '__main__':
    label_ground_truth_masks()
    ground_truth_annotate()
