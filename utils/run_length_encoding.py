import pandas as pd
import pathlib
import imageio
import numpy as np
import skimage

import imaging

from scipy import ndimage
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])


def rle_image(labels_image, image_id):
    '''
    take a labelled image and image id then perform rle and return a pandas dataframe
    '''
    num_labels = np.amax(labels_image)
    df_image = pd.DataFrame(columns=['ImageId','EncodedPixels'])
    for label_num in range(1, num_labels+1):
        label_mask = np.where(labels_image == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            rle_series = pd.Series({'ImageId': image_id[:-4], 'EncodedPixels': rle})
            df_image = df_image.append(rle_series, ignore_index=True)
    return df_image


def rle_images_in_dir(file_path):
    '''
    performs rle on all images in a directory given by file_path
    '''
    image_ids = imaging.get_image_ids(file_path)
    output_path = imaging.get_path('output')

    df_all = pd.DataFrame()
    for idx, image_id in enumerate(image_ids):
        image_dir = file_path + image_id
        image = skimage.io.imread(image_dir)
        df_image = rle_image(image, image_id)
        df_all = df_all.append(df_image, ignore_index=True)
        print('encoded image %d of %d, image: %s \n' % \
             (idx + 1, len(image_ids), image_id[:-4]))
    return df_all

if __name__ == '__main__':
    df = rle_images_in_dir(imaging.get_path('output') + 'labelled_segmented/')
    df.to_csv('submission.csv', index=None)
