import numpy as np
import skimage.io
import matplotlib.pyplot as plt 
import skimage.segmentation
import os
import sys

# add a single image and its associated masks
id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
gt = "../output/labelled_ground_truth/{}.png".format(id)
segmented = "../output/labelled_segmented/{}.png".format(id)



gt = skimage.io.imread(gt)
segmented = skimage.io.imread(segmented)


# Compute number of objects
true_objects = len(np.unique(gt))
pred_objects = len(np.unique(segmented))
print("Number of true objects:", true_objects)
print("Number of predicted objects:", pred_objects)


# Compute intersection between all objects
intersection = np.histogram2d(gt.flatten(), segmented.flatten(), bins=(true_objects, pred_objects))[0]


# Compute areas (needed for finding the union between all objects)
area_true = np.histogram(gt, bins = true_objects)[0]
area_pred = np.histogram(segmented, bins = pred_objects)[0]
area_true = np.expand_dims(area_true, -1)
area_pred = np.expand_dims(area_pred, 0)


union = area_true + area_pred - intersection


# Exclude background from the analysis
intersection = intersection[1:,1:]
union = union[1:,1:]
union[union == 0] = 1e-9

# Compute the intersection over union
iou = intersection / union


# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

prec = []
print("Thresh\tTP\tFP\tFN\tPrec.")
for t in np.arange(0.5, 1.0, 0.05):
    tp, fp, fn = precision_at(t, iou)
    p = tp / (tp + fp + fn)
    print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
    prec.append(p)
print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
