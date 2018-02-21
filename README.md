# Nuclei Segmentation
This repository is a framework for submission for the 2018 Data Science Bowl competition to segment nuclei in cell microscopy images. Identifying cell nuclei is the starting point for many biological analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. Segmenting nuclei allows researchers to identify each individual cell in a sample, measure morphometry, and analyse how cells react to various treatments.

The objective is to create a single model for segmentation that works across all kinds of image modalities, no matter the size of the nuclei or the image color scheme. Such a model could be built into software that biologists use with all kinds of microscopes and eliminate the need for them to train on their individual data or provide metadata about their cell type, microscope, or resolution.

## Images
![alt text](./output/stage1/train/montage_train.png)


## Evaluation
![Evaluation of this competition](https://www.kaggle.com/c/data-science-bowl-2018#evaluation) using the mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated.

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value `t`, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects.

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold.

Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.

We have implemented a version of this evaluation metric for offline in evaluation of the training data in the `./utils/evaluate.py` file.

## Run-length encoding
![Run-length encoding](https://www.kaggle.com/c/data-science-bowl-2018#evaluation) is using for competition submissions to reduce the submission file size. Instead of submitting an exhaustive list of indices for your segmentation, pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

We have implemented a method to perform run-length encoding here `./utils/evaluate.py` file.

## Pipeline

[video](https://datasciencebowl.com/2018dsbtutorial/video)
[DataScienceBowl.com](https://datasciencebowl.com/)

Blogs:
https://medium.com/stanford-ai-for-healthcare/how-different-are-cats-and-cells-anyway-closing-the-gap-for-deep-learning-in-histopathology-14f92d0ddb63

Papers:
	
