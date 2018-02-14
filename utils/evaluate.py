import numpy as np
import skimage.io
import skimage.segmentation
import os
import sys
import utils.imaging as imaging

from utils.imaging import get_path, get_image_ids

def calculate_iou(ground_truth, segmented):
    # calculate number of objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(segmented))
    print("# true nuclei:", true_objects - 1)
    print("# predicted pred:", pred_objects - 1)

    # compute intersection between all objects
    intersection = np.histogram2d(ground_truth.flatten(), segmented.flatten(),\
                   bins=(true_objects, pred_objects))[0]

    # compute areas needed for finding the union between objects
    area_true = np.histogram(ground_truth, bins = true_objects)[0]
    area_pred = np.histogram(segmented, bins = pred_objects)[0]
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
    # precision helper function
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1
    false_positives = np.sum(matches, axis=0) == 0
    false_negatives = np.sum(matches, axis=1) == 0
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), \
                        np.sum(false_negatives)
    return tp, fp, fn

def evaluate_image(ground_truth, segmented):
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

def evaluate_images(stage_num = 1):
    stage_num = str(stage_num)
    file_path = get_path('data_train_' + stage_num)
    image_ids = get_image_ids(file_path)
    label_ground_truth = get_path('output_train_' + stage_num + '_lab_gt')
    label_segmented = get_path('output_train_' + stage_num + '_lab_seg')

    mean_score = []
    for idx, image_id in enumerate(image_ids):
        ground_truth_path = label_ground_truth + image_id + ".png"
        ground_truth = skimage.io.imread(ground_truth_path)
        segmented_path = label_segmented + image_id + ".png"
        segmented = skimage.io.imread(segmented_path)
        score = evaluate_image(ground_truth, segmented)
        mean_score.append(score)
        print("image: " + str(idx) + " of " + str(len(image_ids)) + \
              "\n" + str(image_id) + "\nscore is " + str(score) + "\n")
    #mean_score = np.mean(mean_score)
    return mean_score

if __name__ == '__main__':
    score = evaluate_images()
    print("mean total score: " + str(score))
