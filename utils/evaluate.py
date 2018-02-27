import os
import sys
import configparser

import skimage.io

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K 
from utils.imaging import get_path, get_image_ids


def calc_intersection(ground_truth,segmented):
    """Calculate the intersection of two flattened arrays using histogram2d

    Args:
        ground_truth (np.array): The ground truth image array.
        segmented (np.array): The segmented image array.

    Returns:
        ndarray: The bi-dimensional histogram of samples

    """
    return np.histogram2d(ground_truth.flatten(), segmented.flatten(),\
               bins=(len(np.unique(ground_truth)), len(np.unique(segmented))))[0]

def obj_count(mask):
    """ Compute the pixel count of each object in the mask image.

    Args:
        mask (np.array): Boolean array of objects.

    Returns:
        ndarray: The values of the histogram.

    """
    return np.histogram(mask, bins = len(np.unique(mask)))[0]

def calculate_iou(ground_truth, segmented):
    """ Calculate the intersection over the union for two images
    https://www.kaggle.com/wcukierski/example-metric-implementation

    Args:
        mask (np.array): Boolean array of objects.

    Returns:
        iou (float): intersection over union score.

    """
    # calculate number of objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(segmented))
    print("# true nuclei:", true_objects - 1)
    print("# predicted pred:", pred_objects - 1)

    # computer the intersection between all object
    intersection = calc_intersection(ground_truth,segmented)
    # compute areas needed for finding the union between objects
    area_true = obj_count(ground_truth)
    area_pred = obj_count(segmented)
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection

    # exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # compute the intersection over union
    iou = intersection / union
    return iou

def precision_at_iou(threshold, iou):
    """ Calculate the precision at an iou value

    Args:
        threshold (float): precision thresholds e.g. 0.50, 0.55, 0.60, 0.65, ...

    Returns:
        tp (int): number of true positives.
        fp (int): number of false positives.
        fn (int): number of false negatives.

    """
    # precision helper function
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1
    false_positives = np.sum(matches, axis=0) == 0
    false_negatives = np.sum(matches, axis=1) == 0
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), \
                        np.sum(false_negatives)
    return tp, fp, fn

def evaluate_image(ground_truth, segmented):
    """ Evaluate the average precision fo an image

    Args:
        ground_truth (np.array): The ground truth image array.
        segmented (np.array): The segmented image array.

    Returns:
        score (float): mean score for image.

    """
    iou = calculate_iou(ground_truth, segmented)
    prec = []

    print("\n{}\t{}\t{}\t{}\t{}".format('thresh', 'tp', 'fp', 'fn', 'p'))
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at_iou(t, iou)
        p = tp / (tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    score = np.mean(prec)
    return score

def ap(y_true, y_pred):
    '''
    alternative ap calculation
    https://www.kaggle.com/thomasjpfan/ap-metric
    '''
    # remove one for background
    num_true = len(np.unique(y_true)) - 1
    num_pred = len(np.unique(y_pred)) - 1

    if num_true == 0 and num_pred == 0:
        return 1
    elif num_true == 0 or num_pred == 0:
        return 0

    # bin size + 1 for background
    intersect = np.histogram2d(
        y_true.flatten(), y_pred.flatten(), bins=(num_true+1, num_pred+1))[0]

    area_t = np.histogram(y_true, bins=(num_true+1))[0][:, np.newaxis]
    area_p = np.histogram(y_pred, bins=(num_pred+1))[0][np.newaxis, :]

    # get rid of background
    union = area_t + area_p - intersect
    intersect = intersect[1:, 1:]
    union = union[1:, 1:]
    iou = intersect / union

    threshold = np.arange(0.5, 1.0, 0.05)[np.newaxis, np.newaxis, :]
    matches = iou[:,:, np.newaxis] > threshold

    tp = np.sum(matches, axis=(0,1))
    fp = num_true - tp
    fn = num_pred - tp

    return np.mean(tp/(tp+fp+fn))

def keras_mean_iou(y_true, y_pred):
    """ Evaluate the average precision for an image using Keras

    Args:
        y_true (np.array): The ground truth image array.
        y_pred (np.array): The segmented image array.

    Returns:
        (float): mean score for image.

    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def evaluate_images(stage_num = 1):
    """ Evaluate all images for stage in directory

    Args:
        stage_num (int): Competition stage number.

    Returns:
        scores (dataframe): pandas dataframe of image ids and scores.

    """
    stage_num = str(stage_num)
    file_path = get_path('data_train_' + stage_num)
    image_ids = get_image_ids(file_path)
    label_ground_truth = get_path('output_train_' + stage_num + '_lab_gt')
    label_segmented = get_path('output_train_' + stage_num + '_lab_seg')

    cols = ['image_id', 'score']
    scores = []
    for idx, image_id in enumerate(image_ids):
        ground_truth_path = label_ground_truth + image_id + ".png"
        ground_truth = skimage.io.imread(ground_truth_path)
        segmented_path = label_segmented + image_id + ".png"
        segmented = skimage.io.imread(segmented_path)
        score = evaluate_image(ground_truth, segmented)
        #score = ap(ground_truth, segmented)
        scores.append([image_id, score])
        print("image: " + str(idx) + " of " + str(len(image_ids)) + \
              "\n" + str(image_id) + "\nscore is " + str(score) + "\n")
    df_scores = pd.DataFrame(scores, columns=cols).round(4)
    return df_scores

def submit_kaggle(notebook_name, submission_path, message):
    """ Generate submission string that can be used to submit an entry for scoring by 
    kaggle on test data. Use python magic commands to run 
    
    Args:
        notebook_name (str): Name of jupyter notebook for kaggle entry
        submission_path (str): Path to run length encoded csv data
        message (str): an optional message about data used for submission
    
    Returns:
        submit_string (str): correctly formatted string for kaggle submission
    
    """
    config = configparser.ConfigParser()
    config.read("/home/ubuntu/.kaggle-cli/config")
    comp = config.get('user', 'competition')
    uname = config.get('user', 'username')
    password = config.get('user', 'password')
    submit_string = 'kg submit {} -u \"{}\" -p \"{}\" -c \"{}\" -m \"{}: {}\"' \
    .format(submission_path, uname, password, comp, notebook_name, message)
    return submit_string

if __name__ == '__main__':
    score = evaluate_images()
    print("mean total score: " + str(score))
