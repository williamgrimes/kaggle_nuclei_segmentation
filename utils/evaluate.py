import numpy as np
import skimage.io
import skimage.segmentation
import os
import sys
import imaging

def calculate_iou(ground_truth, segmented):
    # calculate number of objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(segmented))
    print("# true nuclei:", true_objects)
    print("# predicted pred:", pred_objects)

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
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at_iou(t, iou)
        p = tp / (tp + fp + fn)
        prec.append(p)
    score = np.mean(prec)
    return score

def evaluate_images():
    file_path = imaging.get_training_data_path()
    image_ids = imaging.get_image_ids(file_path)
    output_path = imaging.get_output_path()
    mean_score = []
    for idx, image_id in enumerate(image_ids):
        ground_truth_path = output_path + "/labelled_ground_truth/" + \
                            image_id + ".png"
        ground_truth = skimage.io.imread(ground_truth_path)
        segmented_path = output_path + "/labelled_segmented/" + \
                         image_id + ".png"
        segmented = skimage.io.imread(segmented_path)
        score = evaluate_image(ground_truth, segmented)
        mean_score.append(score)
        print("image: " + str(idx) + " of " + str(len(image_ids)) + \
              "\n" + str(image_id) + "\nscore is " + str(score) + "\n")
    mean_score = np.mean(mean_score)
    return mean_score

if __name__ == '__main__':
    score = evaluate_images()
    print("mean total score: " + str(score))
