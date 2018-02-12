import os
import glob
import skimage

import numpy as np
from scipy import ndimage
from skimage.color import rgba2rgb
from skimage.io import imread, imsave
from skimage.segmentation import mark_boundaries


def get_path(dir_type='project'):
    if dir_type == 'project':
        return os.environ['NUC_SEG_DIR']
    if dir_type == 'data':
        return os.environ['NUC_SEG_DIR'] + '/data/'
    if dir_type == 'training_data':
        return os.environ['NUC_SEG_DIR'] + '/data/stage1_train/'
    if dir_type == 'envs':
        return os.environ['NUC_SEG_DIR'] + '/data/envs/'
    if dir_type == 'output':
        return os.environ['NUC_SEG_DIR'] + '/output/'
    if dir_type == 'models':
        return os.environ['NUC_SEG_DIR'] + '/models/'

def get_image_ids(path):
    "returns a list of images at path to train or test data"
    image_ids = sorted([f for f in os.listdir(path) \
                        if not f.startswith('.')])
    return image_ids

def label_mask(mask):
    ''' mask out background extract connected objects and label'''
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    return labels

def label_ground_truth_masks():
    ''' labels masks of training data and saves'''
    train_data_dir = get_path('training_data')
    image_ids = get_image_ids(train_data_dir)
    output_path = get_path('output') +  "labelled_ground_truth/"

    for idx, image_id in enumerate(image_ids):
        masks = train_data_dir + image_id +  "/masks/*.png"
        masks = skimage.io.imread_collection(masks).concatenate()

        num_masks, height, width = masks.shape

        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1

        imsave(output_path + image_id + '.png', labels)

        print('saved image %d of %d, image: %s \n' % \
              (idx + 1, len(image_ids), image_id))

def ground_truth_annotate(file_path):
    "annotates images by showing ground truth contours from masks"
    image_ids = get_image_ids(file_path)
    output_path = get_path('output') + "annotated_ground_truth/"

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

        imsave(output_path + image_id + '.png', image_overlay)

        print('saved image %d of %d, image: %s \n' % \
              (idx + 1, len(image_ids), image_id))

if __name__ == '__main__':
    label_ground_truth_masks()
    ground_truth_annotate(get_path('training_data'))
