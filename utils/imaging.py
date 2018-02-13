import os
import glob
import skimage

import numpy as np
from scipy import ndimage, misc
from skimage.color import rgba2rgb
from skimage.io import imread, imsave
from skimage.segmentation import mark_boundaries


def get_path(dir_type='project'):
    '''
    A function to get relevant project directories

    Arguments:
        dir_type e.g. project/data/training_data/test...

    Returns:
        path to directory as a string
    '''
    if dir_type == 'project':
        return os.environ['NUC_SEG_DIR']
    if dir_type == 'data':
        return os.environ['NUC_SEG_DIR'] + '/data/'
    if dir_type == 'training_data':
        return os.environ['NUC_SEG_DIR'] + '/data/stage1_train/'
    if dir_type == 'test_data':
        return os.environ['NUC_SEG_DIR'] + '/data/stage1_test/'
    if dir_type == 'envs':
        return os.environ['NUC_SEG_DIR'] + '/data/envs/'
    if dir_type == 'output':
        return os.environ['NUC_SEG_DIR'] + '/output/'
    if dir_type == 'models':
        return os.environ['NUC_SEG_DIR'] + '/models/'

def get_image_ids(path):
    '''
    A function to get list of image ids from a directory containing id files this excludes dotfiles

    Arguments:
        path to directory

    Returns:
        list of image ids
    '''
    image_ids = sorted([f for f in os.listdir(path) \
                        if not f.startswith('.')])
    return image_ids

def label_mask(mask):
    '''
    Uses ndimage function to labels connected components in a binary image mask with sequential values 

    Arguments:
        mask as numpy array

    Returns:
        array of labels as numpy array
    '''
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    return labels

def label_ground_truth_masks():
    '''
    Iterates over ground truth training masks and labels saving output labelled images
    '''
    train_data_dir = get_path('training_data')
    image_ids = get_image_ids(train_data_dir)
    output_path = get_path('output') +  "labelled_ground_truth/"

    for idx, image_id in enumerate(image_ids):
        masks = train_data_dir + image_id +  "/masks/*.png"
        masks = imread_collection(masks).concatenate()

        num_masks, height, width = masks.shape

        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1

        imsave(output_path + image_id + '.png', labels)

        print('saved image %d of %d, image: %s \n' % \
              (idx + 1, len(image_ids), image_id))

def ground_truth_annotate(file_path):
    '''
    Annotates by marking boundaries in training ground truth images from masks, and saves output
    '''
    image_ids = get_image_ids(file_path)
    output_path = get_path('output') + "train/annotated_ground_truth/"

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
        print_tuple = (idx + 1, len(image_ids), image_id)
        print('saved image %d of %d, image: %s \n' % print_tuple)


def segmented_annotate(image_type='train'):
    '''
    Annotates by marking boundaries in segmented images from labelled images, and saves output
    '''
    if image_type == 'train':
        file_path = get_path('data') + 'stage1_train/'
        image_ids = get_image_ids(file_path)
    if image_type == 'test':
        file_path = get_path('data') + 'stage1_test/'
        image_ids = get_image_ids(file_path)

    input_path = get_path('output') + image_type + '/labelled_segmented/'
    output_path = get_path('output') + image_type + '/annotated_segmented/'

    for idx, image_id in enumerate(image_ids):
        image_dir = file_path + image_id + '/images/' + \
                    image_id + '.png'
        image = imread(image_dir)
        if image.shape[2] == 4:
            image = rgba2rgb(image)

        labelled_dir = input_path + image_id + '.png'
        labelled_image = misc.imread(labelled_dir)

        image_overlay = image
        for i in range(1, labelled_image.max()+1):
            mask = (labelled_image==i)
            image_overlay = mark_boundaries(image_overlay,
                                        mask, color=(0.6,0,0),
                                        outline_color=None,
                                        mode='outer')

        imsave(output_path + image_id + '.png', image_overlay)
        print_tuple = (idx + 1, len(image_ids), image_id)
        print('saved image %d of %d, image: %s \n' % print_tuple)


if __name__ == '__main__':
    label_ground_truth_masks()
    ground_truth_annotate(get_path('training_data'))
    segmented_annotate('train')
    segmented_annotate('test')
