import pandas as pd
import pathlib
import imageio
import numpy as np

import imaging

from scipy import ndimage
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])


def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings
    and dump it into a Pandas DataFrame.
    '''
    # Read in data and convert to grayscale
    im_id = im_path.parts[-3]
    im = imageio.imread(str(im_path))
    im_gray = rgb2gray(im)

    # Mask out background and extract connected objects
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)

    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame(columns=['ImageId','EncodedPixels'])
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': im_id, 'EncodedPixels': rle})
            im_df = im_df.append(s, ignore_index=True)
    return im_df


def analyze_list_of_images(im_path_list):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for idx, im_path in enumerate(im_path_list):
        im_df = analyze_image(im_path)
        all_df = all_df.append(im_df, ignore_index=True)
        print('encoded image %d of %d, image: %s \n' % \
             (idx + 1, len(im_path_list), im_path))
    return all_df

if __name__ == '__main__':
    filepath = imaging.get_training_data_path()
    print(filepath)
    training = pathlib.Path(filepath).glob('*/images/*.png')
    df = analyze_list_of_images(list(training))
    df.to_csv('submission.csv', index=None)
    #for i in training:
        #print(i)
