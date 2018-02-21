import os
import glob
import skimage

import numpy as np
from scipy import ndimage, misc
from skimage.color import rgba2rgb
from skimage.io import imread, imsave, imread_collection
from skimage.segmentation import mark_boundaries

def get_path(directory='project'):
    '''
    A function to get relevant project directories, this relies on having
    exported the environment variables, adding the project to the python 
    path.

    Args:
        directory (str): key for directories dictionary below.

    Returns:
        path (str): path to directory

    '''
    project_dir = os.environ['NUC_SEG_DIR']
    dir_dict = {'project' : '/',
                'data' : '/data/',
                'data_train_1' : '/data/stage1_train/',
                'data_test_1' : '/data/stage1_test/',
                'data_train_2' : '/data/stage2_train/',
                'data_test_2' : '/data/stage2_test/',
                'envs' : '/data/envs/',
                'output' : '/output/',
                'models' : '/models/',
                'output_train_1' : '/output/stage1/train/',
                'output_train_1_ann_gt' : '/output/stage1/train/annotated_ground_truth/',
                'output_train_1_ann_seg' : '/output/stage1/train/annotated_segmented/',
                'output_train_1_lab_gt' : '/output/stage1/train/labelled_ground_truth/',
                'output_train_1_lab_seg' : '/output/stage1/train/labelled_segmented/',
                'output_test_1' : '/output/stage1/test/',
                'output_test_1_ann_seg' : '/output/stage1/test/annotated_segmented/',
                'output_test_1_lab_seg' : '/output/stage1/test/labelled_segmented/',
                'output_train_2' : '/output/stage2/train/',
                'output_train_2_ann_gt' : '/output/stage2/train/annotated_ground_truth/',
                'output_train_2_ann_seg ' : '/output/stage2/train/annotated_segmented/',
                'output_train_2_lab_gt' : '/output/stage2/train/labelled_ground_truth/',
                'output_train_2_lab_seg' : '/output/stage2/train/labelled_segmented/',
                'output_test_2' : '/output/stage2/test/',
                'output_test_2_ann_seg' : '/output/stage2/test/annotated_segmented/',
                'output_test_2_lab_seg' : '/output/stage2/test/labelled_segmented/'
                }
    path = project_dir + dir_dict[directory]
    return path

def get_image_ids(path):
    '''
    A function to get list of image ids from a directory containing id files this excludes dotfiles

    Arguments:
        path (str): path to directory

    Returns:
        image_ids (list): list of image ids

    '''
    image_ids = sorted([f for f in os.listdir(path) \
                        if not f.startswith('.')])
    return image_ids

def label_mask(mask):
    '''
    Uses ndimage function to labels connected components in a binary image mask with sequential values 

    Arguments:
        mask (ndarray): mask image as numpy array

    Returns:
        labels (ndarray): array of labels as numpy array

    '''
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    return labels

def label_ground_truth_masks(stage_num = 1):
    '''
    Iterates over ground truth training masks and labels saving output labelled images

    Arguments:
        stage_num (int): 1 or 2 depending on whether stage1 or stage2

    '''
    stage_num = str(stage_num)
    train_data_dir = get_path('data_train_' + stage_num)
    image_ids = get_image_ids(train_data_dir)
    output_path = get_path('output_train_' + stage_num + '_lab_gt')

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

def ground_truth_annotate(stage_num = 1):
    '''
    Annotates by marking boundaries in training ground truth images from masks, and saves output.

    '''
    stage_num = str(stage_num)
    train_data_dir = get_path('data_train_' + stage_num)
    image_ids = get_image_ids(train_data_dir)
    output_path = get_path('output_train_' + stage_num + '_ann_gt')

    for idx, image_id in enumerate(image_ids):
        image_dir = train_data_dir + image_id + "/images/" + \
                    image_id + ".png"
        image = skimage.io.imread(image_dir)
        image = rgba2rgb(image)

        masks = os.listdir(train_data_dir + image_id + "/masks/")

        image_overlay = image
        for mask in masks:
            mask_name = train_data_dir + image_id + "/masks/" + mask
            mask = skimage.io.imread(mask_name)
            image_overlay = mark_boundaries(image_overlay,
                                            mask, color=(0.6,0,0),
                                            outline_color=None,
                                            mode='outer')

        imsave(output_path + image_id + '.png', image_overlay)
        print_tuple = (idx + 1, len(image_ids), image_id)
        print('saved image %d of %d, image: %s \n' % print_tuple)


def segmented_annotate(image_type='train', stage_num = 1):
    '''
    Annotates by marking boundaries in segmented images from labelled images, and saves output

    Args:
        image_type: 'train' or 'test
        stage_num: stage number 1 or 2

    '''
    stage_num = str(stage_num)
    file_path = get_path('data_' + image_type + '_' + stage_num)
    image_ids = get_image_ids(file_path)
    input_path = get_path('output_' + image_type + '_' + stage_num + '_lab_seg')
    output_path = get_path('output_' + image_type + '_' + stage_num + '_ann_seg')

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
    label_ground_truth_masks(1)
    ground_truth_annotate(1)
    segmented_annotate(image_type = 'train', stage_num = 1)
    segmented_annotate(image_type = 'test', stage_num = 1)
