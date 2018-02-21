# Nuclei Segmentation
This repository is a framework for submission for the 2018 Data Science Bowl competition to segment nuclei in cell microscopy images. Identifying cell nuclei is the starting point for many biological analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. Segmenting nuclei allows researchers to identify each individual cell in a sample, measure morphometry, and analyse how cells react to various treatments.

The objective of this competition is to create a single generalised model for segmentation that works across all kinds of microscopy and image modalities, no matter the size of the nuclei or the image color scheme. Such a model could be built into software that biologists use with all kinds of microscopes and eliminate the need for them to train on their individual data or provide metadata about their cell type, microscope, or resolution.

## Images
Below is a montage of images from the stage 1 test data to show the difference in image modalities that the model should perform on:
![alt text](./output/stage1/test/montage_test.png)

## Running a pipeline:
In `main.py` an example run through of the project pipeline is given by first converting images to greyscale, using an [Otsu threshold](https://en.wikipedia.org/wiki/Otsu%27s_method) to segment greyscale nuclei images, then removing connected components less than 20 pixels.

The `main.py` script does the following:
1. Applies the above described segmentation method to train images and saves a labelled image. 
2. Applies the above described segmentation method to test images and saves a labelled image. 
3. Creates a labelled image of ground truth masks, for comparison with the segmented labelled images. 
4. Annotates ground truth images by outlining segmentation contours on the original image.
4. Annotates segmented training images.
5. Annotates segmented test images.
6. Runs the evaluation metric on the labelled ground truth against the labelled segmented images, and creates a dataframe with the results.
7. Creates a run-length encoded `rle_submission.csv` file for submission of test segmentation to kaggle.

This describes the pipeline for a complete iteration of kaggle submission. Critically, the area to be developed is the modelling that will use the ground truth masks as training data and output, labelled segmented images. These labelled segmented images will be evaluated against the ground truth labelled images and submitted to kaggle, on each iteration new annotations can be generated to show where the model is performing well and where it could be improved.

For posterity trained models should be pickled in a format with initials and date in the `./models/` folder, so the work is reproducible a a notebook showing how the model was generated should also be put in the `./notebooks/` folder.

## Model evaluation
[Evaluation of this competition](https://www.kaggle.com/c/data-science-bowl-2018#evaluation) uses the mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated.

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value `t`, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects.

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold.

Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.

We have implemented a version of this evaluation metric for offline in evaluation of the training data in the `./utils/evaluate.py` file.

## Submission: run-length encoding
[Run-length encoding](https://www.kaggle.com/c/data-science-bowl-2018#evaluation) is using for competition submissions to reduce the submission file size. Instead of submitting an exhaustive list of indices for your segmentation, pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

We have implemented a method to perform run-length encoding here `./utils/evaluate.py` file.

## Resources:
* [DataScienceBowlVideo](https://datasciencebowl.com/2018dsbtutorial/video)
* [DataScienceBowl.com](https://datasciencebowl.com/)
* [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)
* [U-Net paper](https://arxiv.org/abs/1505.04597)
